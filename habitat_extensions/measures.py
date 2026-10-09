import gzip
import json
import pickle
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from dtw import dtw
from fastdtw import fastdtw
from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Action, EmbodiedTask, Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.nav import DistanceToGoal, Success
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import fog_of_war
from habitat.utils.visualizations import maps as habitat_maps
from numpy import ndarray

from habitat_extensions import maps

cv2 = try_cv2_import()


def euclidean_distance(pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


@registry.register_measure
class PathLength(Measure):
    """Path Length (PL)
    PL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    cls_uuid: str = "path_length"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(current_position, self._previous_position)
        self._previous_position = current_position


@registry.register_measure
class OracleNavigationError(Measure):
    """Oracle Navigation Error (ONE)
    ONE = min(geosdesic_distance(agent_pos, goal)) over all points in the
    agent path.
    """

    cls_uuid: str = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        distance_to_target = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class OracleSuccess(Measure):
    """Oracle Success Rate (OSR). OSR = I(ONE <= goal_radius)"""

    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Config, **kwargs: Any):
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        d = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(self._metric or d < self._config.SUCCESS_DISTANCE)


@registry.register_measure
class OracleSPL(Measure):
    """OracleSPL (Oracle Success weighted by Path Length)
    OracleSPL = max(SPL) over all points in the agent path.
    """

    cls_uuid: str = "oracle_spl"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, ["spl"])
        self._metric = 0.0

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        spl = task.measurements.measures["spl"].get_metric()
        self._metric = max(self._metric, spl)


@registry.register_measure
class StepsTaken(Measure):
    """Counts the number of times update_metric() is called. This is equal to
    the number of times that the agent takes an action. STOP counts as an
    action.
    """

    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric += 1.0


@registry.register_measure
class CorrectSignFollowCount(Measure):
    """Counts GT-route signs passed before the agent leaves the GT path.

    A sign is considered correctly followed once the agent has progressed a
    short distance beyond that sign's nearest point on the reference path while
    remaining close to the reference path. If the agent moves off the reference
    path first, the count is frozen.
    """

    cls_uuid: str = "correct_sign_follow_count"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._off_path_distance  = self._config.OFF_PATH_DISTANCE
        self._sign_pass_distance = self._config.SIGN_PASS_DISTANCE
        self._backtrack_distance = self._config.BACKTRACK_DISTANCE
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    @staticmethod
    def _xz(position: Union[List[float], ndarray]) -> ndarray:
        position = np.asarray(position, dtype=np.float32)
        if position.shape[0] == 2:
            return position
        return np.array([position[0], position[2]], dtype=np.float32)

    def _nearest_path_progress(self, position: Union[List[float], ndarray]) -> Tuple[float, float]:
        point = self._xz(position)

        if self._path_xz is None or len(self._path_xz) == 0:
            return 0.0, float("inf")

        if len(self._path_xz) == 1:
            return 0.0, float(np.linalg.norm(point - self._path_xz[0]))

        best_progress = 0.0
        best_distance = float("inf")
        for index in range(len(self._path_xz) - 1):
            seg_start = self._path_xz[index]
            seg_end = self._path_xz[index + 1]
            seg_vec = seg_end - seg_start
            seg_len_sq = float(np.dot(seg_vec, seg_vec))

            if seg_len_sq <= 1e-8:
                candidate = seg_start
                t = 0.0
            else:
                t = float(np.dot(point - seg_start, seg_vec) / seg_len_sq)
                t = float(np.clip(t, 0.0, 1.0))
                candidate = seg_start + t * seg_vec

            distance = float(np.linalg.norm(point - candidate))
            if distance < best_distance:
                best_distance = distance
                best_progress = float(self._cum_path_lengths[index] + t * self._seg_lengths[index])

        return best_progress, best_distance

    @staticmethod
    def _hint_center(hint_bbox_3d: dict) -> Optional[List[float]]:
        if not hint_bbox_3d:
            return None

        bbox_3d = list(hint_bbox_3d.values())[0]
        if not bbox_3d:
            return None

        return bbox_3d[0]

    def _build_sign_progresses(self, episode) -> List[float]:
        sign_progresses = []
        for hint_bbox_3d in getattr(episode, "hints_bbox_3d", []) or []:
            center = self._hint_center(hint_bbox_3d)
            if center is None:
                continue

            progress, _ = self._nearest_path_progress(center)
            sign_progresses.append(progress)

        return sorted(sign_progresses)

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        reference_path = getattr(episode, "reference_path", None) or []
        self._path_xz = np.array([[p[0], p[2]] for p in reference_path], dtype=np.float32)

        if len(self._path_xz) > 1:
            self._seg_lengths = np.linalg.norm(self._path_xz[1:] - self._path_xz[:-1], axis=1)
            self._cum_path_lengths = np.concatenate([[0.0], np.cumsum(self._seg_lengths)])
        else:
            self._seg_lengths = np.array([], dtype=np.float32)
            self._cum_path_lengths = np.array([0.0], dtype=np.float32)

        self._sign_progresses = self._build_sign_progresses(episode)
        self._next_sign_index = 0
        self._off_path = False
        self._correct_sign_count = 0.0
        self._metric = 0.0

        self._previous_progress, self._previous_path_distance = self._nearest_path_progress(
            self._sim.get_agent_state().position
        )

        self._update_ratio_metric()

    def _update_ratio_metric(self):
        total_signs = len(self._sign_progresses)
        self._metric = self._correct_sign_count / total_signs if total_signs > 0 else 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        if self._off_path or len(self._sign_progresses) == 0:
            return

        current_progress, current_path_distance = self._nearest_path_progress(
            self._sim.get_agent_state().position
        )

        if current_path_distance > self._off_path_distance:
            self._off_path = True
            return

        if current_progress + self._backtrack_distance < self._previous_progress:
            self._off_path = True
            return

        while self._next_sign_index < len(self._sign_progresses):
            sign_progress = self._sign_progresses[self._next_sign_index]
            if current_progress < sign_progress + self._sign_pass_distance:
                break

            self._correct_sign_count += 1.0
            self._next_sign_index += 1

        self._update_ratio_metric()
        self._previous_progress = current_progress
        self._previous_path_distance = current_path_distance


@registry.register_measure
class NDTW(Measure):
    """NDTW (Normalized Dynamic Time Warping)
    ref: https://arxiv.org/abs/1907.05446
    """

    cls_uuid: str = "ndtw"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.dtw_func = fastdtw if config.FDTW else dtw

        with gzip.open(config.GT_PATH.format(split=config.SPLIT), "rt") as f:
            self.gt_json = json.load(f)

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.locations = []
        self.gt_locations = self.gt_json[episode.episode_id]["locations"]
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position == self.locations[-1]:
                return
            self.locations.append(current_position)

        dtw_distance = self.dtw_func(self.locations, self.gt_locations, dist=euclidean_distance)[0]

        nDTW = np.exp(-dtw_distance / (len(self.gt_locations) * self._config.SUCCESS_DISTANCE))
        self._metric = nDTW


@registry.register_measure
class SDTW(Measure):
    """SDTW (Success Weighted be nDTW)
    ref: https://arxiv.org/abs/1907.05446
    """

    cls_uuid: str = "sdtw"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [NDTW.cls_uuid, Success.cls_uuid])
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        nDTW = task.measurements.measures[NDTW.cls_uuid].get_metric()
        self._metric = ep_success * nDTW


@registry.register_measure
class TopDownMapVLNCE(Measure):
    """A top down map that optionally shows VLN-related visual information
    such as MP3D node locations and MP3D agent traversals.
    """

    cls_uuid: str = "top_down_map_vlnce"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._step_count = None
        self._map_resolution = config.MAP_RESOLUTION
        self._previous_xy_location = None
        self._top_down_map = None
        self._meters_per_pixel = None
        self.current_node = ""
        with open(self._config.GRAPHS_FILE, "rb") as f:
            self._conn_graphs = pickle.load(f)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_original_map(self) -> ndarray:
        top_down_map = habitat_maps.get_topdown_map_from_sim(self._sim, 
                                                             map_resolution=self._map_resolution,
                                                             draw_border=self._config.DRAW_BORDER,
                                                             meters_per_pixel=self._meters_per_pixel)

        self._fog_of_war_mask = None
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)

        return top_down_map

    def reset_metric(self, *args: Any, episode: Episode, **kwargs: Any) -> None:
        self._scene_id = episode.scene_id.split("/")[-2]
        self._step_count = -1 # default: 0
        self._metric = None
        self._meters_per_pixel = habitat_maps.calculate_meters_per_pixel(self._map_resolution, self._sim)
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        scene_id = episode.scene_id.split("/")[-1].split(".")[0]
        a_x, a_y = habitat_maps.to_grid(agent_position[2], agent_position[0], self._top_down_map.shape[0:2], sim=self._sim)
        self._previous_xy_location = (a_y, a_x)

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                np.array([a_x, a_y]),
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST / habitat_maps.calculate_meters_per_pixel(self._map_resolution, sim=self._sim))

        if self._config.DRAW_FIXED_WAYPOINTS:
            maps.draw_mp3d_nodes(self._top_down_map, self._sim, episode, self._conn_graphs[scene_id], self._meters_per_pixel)

        if self._config.DRAW_SHORTEST_PATH:
            shortest_path_points = self._sim.get_straight_shortest_path_points(agent_position, episode.goals[0].position)
            maps.draw_straight_shortest_path_points(self._top_down_map, self._sim, self._map_resolution, shortest_path_points)

        if self._config.DRAW_REFERENCE_PATH:
            maps.draw_reference_path(self._top_down_map, self._sim, episode, self._map_resolution, self._meters_per_pixel)

        # draw source and target points last to avoid overlap
        if self._config.DRAW_SOURCE_AND_TARGET:
            maps.draw_source_and_target(self._top_down_map, self._sim, episode, self._meters_per_pixel)

        # MP3D START NODE
        self._nearest_node = maps.get_nearest_node(self._conn_graphs[scene_id], np.take(agent_position, (0, 2)))
        nn_position = self._conn_graphs[self._scene_id].nodes[self._nearest_node]["position"]
        self.s_x, self.s_y = habitat_maps.to_grid(nn_position[2], nn_position[0], self._top_down_map.shape[0:2], self._sim)
        
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        self._step_count += 1
        (house_map, map_agent_pos) = self.update_map(self._sim.get_agent_state().position)

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": map_agent_pos,
            "agent_angle": self.get_polar_angle(),
            "bounds": {k: v for k, v in zip(["lower", "upper"], self._sim.pathfinder.get_bounds())},
            "meters_per_px": self._meters_per_pixel,
        }

    def get_polar_angle(self) -> float:
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(ref_rotation.inverse(), np.array([0, 0, -1]))

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position: List[float]) -> None:
        a_x, a_y = habitat_maps.to_grid(agent_position[2], agent_position[0], self._top_down_map.shape[0:2], self._sim)
        
        gradient_color = 20 
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            maps.drawline(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                gradient_color,
                thickness=int(self._map_resolution * 1.0 / maps.MAP_THICKNESS_SCALAR),
                style="filled",
            )

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                np.array([a_x, a_y]),
                self.get_polar_angle(),
                self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                / habitat_maps.calculate_meters_per_pixel(self._map_resolution, sim=self._sim),
            )

        point_padding = int(0.2 / self._meters_per_pixel)
        prev_nearest_node = self._nearest_node
        self._nearest_node = maps.update_nearest_node(self._conn_graphs[self._scene_id], self._nearest_node, np.take(agent_position, (0, 2)))
        
        if (self._nearest_node != prev_nearest_node and self._config.DRAW_MP3D_AGENT_PATH):
            nn_position = self._conn_graphs[self._scene_id].nodes[self._nearest_node]["position"]
            (prev_s_x, prev_s_y) = (self.s_x, self.s_y)
            self.s_x, self.s_y = habitat_maps.to_grid(nn_position[2], nn_position[0], self._top_down_map.shape[0:2], self._sim)

        self._previous_xy_location = (a_y, a_x)
        map_agent_pos = (a_x, a_y)
        
        return self._top_down_map, map_agent_pos
