import cv2
import ipdb
import math
import habitat_sim
import numpy as np
import magnum as mn
from typing import Any
from copy import deepcopy
from numpy import ndarray
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from skimage.draw import line as sk_line
from typing import Any, Dict, List, Optional, Tuple, Union

from gym import Space, spaces
from gym.spaces.box import Box
from habitat.config import Config
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from habitat_extensions.datasets import VLNExtendedEpisode
from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat


@registry.register_sensor
class GlobalGPSSensor(Sensor):
    """Current agent location in global coordinate frame"""
    cls_uuid: str = "global_gps"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._dimensionality = config.DIMENSIONALITY
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=(self._dimensionality,),
            dtype=np.float,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position # [x, y, z]
        if self._dimensionality == 2:
            agent_position = np.array([agent_position[0], agent_position[2]]) # [x, z]

        agent_rotation = self._sim.get_agent_state().rotation # quaternion, in x, y, z, w format
        heading_vector = quaternion_rotate_vector(agent_rotation.inverse(), np.array([0, 0, -1]))
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        yaw = np.array(phi) + z_neg_z_flip
    
        return {"position": agent_position.astype(np.float32), "yaw": yaw.astype(np.float32)}


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    """Relative progress towards goal"""

    cls_uuid: str = "progress"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any) -> float:
        distance_to_target = self._sim.geodesic_distance(self._sim.get_agent_state().position.tolist(), episode.goals[0].position)

        # just in case the agent ends up somewhere it shouldn't
        if not np.isfinite(distance_to_target):
            return np.array([0.0])

        distance_from_start = episode.info["geodesic_distance"]
        return np.array([(distance_from_start - distance_to_target) / distance_from_start])


@registry.register_sensor
class AngleFeaturesSensor(Sensor):
    """Returns a fixed array of features describing relative camera poses based
    on https://arxiv.org/abs/1806.02724. This encodes heading angles but
    assumes a single elevation angle.
    """

    cls_uuid: str = "angle_features"

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        self.cameras = config.CAMERA_NUM
        super().__init__(config)
        orient = [np.pi * 2 / self.cameras * i for i in range(self.cameras)]
        self.angle_features = np.stack([np.array([np.sin(o), np.cos(o), 0.0, 1.0]) for o in orient])

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cameras, 4),
            dtype=np.float,
        )

    def get_observation(self, *args: Any, **kwargs: Any) -> ndarray:
        return deepcopy(self.angle_features)


@registry.register_sensor
class ShortestPathSensor(Sensor):
    """Provides the next action to follow the shortest path to the goal."""

    cls_uuid: str = "shortest_path_sensor"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        super().__init__(config=config)
        cls = ShortestPathFollower
        if config.USE_ORIGINAL_FOLLOWER:
            cls = ShortestPathFollowerCompat
        self.follower = cls(sim, config.GOAL_RADIUS, return_one_hot=False)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        if best_action is None:
            best_action = HabitatSimActions.STOP
        return np.array([best_action])


@registry.register_sensor
class ReferencePathSensor(Sensor):
    """Provides the next action to follow the optimal path to the goal."""

    cls_uuid: str = "reference_path_sensor"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        super().__init__(config=config)

        cls = ShortestPathFollower if config.USE_ORIGINAL_FOLLOWER else ShortestPathFollowerCompat
        self.follower = cls(sim, config.GOAL_RADIUS, return_one_hot=False)

        self._sim = sim
        self.episode_id = "-1" 
        self.waypoint_id = None 

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_next_waypoint(self, path: List[List[float]], agent_position: List[float], arrival_threshold: float=0.5) -> Union[List[float], None]:
        if not path or len(path) < 2:
            return path[-1] if path else None

        closest1 = (float('inf'), -1)
        closest2 = (float('inf'), -1)

        for i, point in enumerate(path):
            dist_sq = (point[0] - agent_position[0])**2 + \
                      (point[1] - agent_position[1])**2 + \
                      (point[2] - agent_position[2])**2
            
            if dist_sq < closest1[0]:
                closest2 = closest1
                closest1 = (dist_sq, i)
            elif dist_sq < closest2[0]:
                closest2 = (dist_sq, i)
                
        dist1 = math.sqrt(closest1[0])
        idx1 = closest1[1]
        
        dist2 = math.sqrt(closest2[0])
        idx2 = closest2[1]
        
        if dist1 < arrival_threshold:
            waypoint_index = idx1 + 1
        else:
            waypoint_index = max(idx1, idx2)
            
        final_index = min(waypoint_index, len(path) - 1)
        waypoint = path[final_index]
        return waypoint

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        reference_path = episode.reference_path
        reference_path_len = len(reference_path)

        # compute the closest waypoint
        agent_position_xz = np.array([agent_position[0], agent_position[2]])
        reference_path_xz = np.array([[waypoint[0], waypoint[2]] for waypoint in reference_path])
        kd_tree = KDTree(reference_path_xz)
        _, closest_waypoint_index = kd_tree.query(agent_position_xz)
        
        waypoint_index = closest_waypoint_index
        waypoint = reference_path[waypoint_index]
        waypoint_distance = math.sqrt((waypoint[0] - agent_position[0])**2 + (waypoint[2] - agent_position[2])**2)
        while waypoint_distance < self.config.GOAL_RADIUS * 3:
            waypoint_index += 1
            if waypoint_index >= reference_path_len - 1:
                waypoint_index = reference_path_len - 1
                waypoint = reference_path[waypoint_index]
                break

            waypoint = reference_path[waypoint_index]
            waypoint_distance = math.sqrt((waypoint[0] - agent_position[0])**2 + (waypoint[2] - agent_position[2])**2)

        best_action = self.follower.get_next_action(waypoint) 
        if best_action is None:
            best_action = HabitatSimActions.STOP

        return np.array([best_action])


@registry.register_sensor
class HintPositionSensor(Sensor):
    cls_uuid: str = "hint_position"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self.hint_position = None
        self.enable_forward_hint_dropout = getattr(config, "ENABLE_FORWARD_HINT_DROPOUT", False)
        self.forward_hint_dropout_prob = float(np.clip(getattr(config, "FORWARD_HINT_DROPOUT_PROB", 0.0), 0.0, 1.0))
        self.enable_close_non_forward_hint_dropout = getattr(config, "ENABLE_CLOSE_NON_FORWARD_HINT_DROPOUT", False)
        self.close_non_forward_hint_distance = max(0.0, float(getattr(config, "CLOSE_NON_FORWARD_HINT_DISTANCE", 1.0)))
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return spaces.Box(low=0, high=max(self.config.RGB_HEIGHT, self.config.RGB_WIDTH), shape=(4,), dtype=np.uint16)

    # project a 3D point into 2D image space for a particular sensor
    def get_2d_point(self, sim, sensor_name, point_3d: mn.Vector3):
        # get the scene render camera and sensor object
        visual_sensor = sim._sensors[sensor_name]
        scene_graph = sim.get_active_scene_graph()
        scene_graph.set_default_render_camera_parameters(visual_sensor._sensor_object)
        render_camera = scene_graph.get_default_render_camera()
        # use the camera and projection matrices to transform the point onto the near plane
        projected_point_3d = render_camera.projection_matrix.transform_point(render_camera.camera_matrix.transform_point(point_3d))
        # convert the 3D near plane point to integer pixel space
        point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
        point_2d = point_2d / render_camera.projection_size()[0]
        point_2d += mn.Vector2(0.5)
        point_2d *= render_camera.viewport
        point_depth = projected_point_3d[2]
        return mn.Vector2i(point_2d), point_depth

    def get_directed_bbox(self, sim, hint_bbox_3d: Dict):
        agent_position = sim.get_agent_state().position
        agent_position[1] = 0.0 

        (direction, bbox_3d), = hint_bbox_3d.items()
        bbox_center_3d = bbox_3d[0]
        bbox_center_3d[1] = 0.0
        bbox_corners_3d = bbox_3d[1: ]

        agent_center_vec = np.array(bbox_center_3d) - np.array(agent_position)
        agent_center_vec_norm = agent_center_vec / np.linalg.norm(agent_center_vec)

        left_corners, right_corners = [], []
        for bbox_corner_3d in bbox_corners_3d:
            bbox_corner_3d_temp = [bbox_corner_3d[0], 0.0, bbox_corner_3d[2]]
            agent_corner_vec = np.array(bbox_corner_3d_temp) - np.array(agent_position)
            agent_corner_vec_norm = agent_corner_vec / np.linalg.norm(agent_corner_vec)

            cross_product = np.cross(agent_center_vec_norm, agent_corner_vec_norm)
            
            if cross_product[1] > 1e-6: # left
                left_corners.append(bbox_corner_3d)
            elif cross_product[1] < -1e-6: # right
                right_corners.append(bbox_corner_3d)

        if len(left_corners) < 2 or len(right_corners) < 2:
            return None
        
        top_left_index = np.argmax(np.array(left_corners)[:, 1]) # 1 means the y axis
        top_left_corner = left_corners[top_left_index]
        bottom_left_index = np.argmin(np.array(left_corners)[:, 1]) # 1 means the y axis
        bottom_left_corner = left_corners[bottom_left_index]
        
        top_right_index = np.argmax(np.array(right_corners)[:, 1]) # 1 means the y axis
        top_right_corner = right_corners[top_right_index]
        bottom_right_index = np.argmin(np.array(right_corners)[:, 1]) # 1 means the y axis
        bottom_right_corner = right_corners[bottom_right_index]

        top_left2right_vec = np.array(top_right_corner) - np.array(top_left_corner)
        top_left2right_1_3 = top_left2right_vec / 3
        top_left2right_2_3 = top_left2right_vec * 2 / 3

        bottom_left2right_vec = np.array(bottom_right_corner) - np.array(bottom_left_corner)
        bottom_left2right_1_3 = bottom_left2right_vec / 3
        bottom_left2right_2_3 = bottom_left2right_vec * 2 / 3

        if direction == "left":
            top_right_corner = top_left_corner + top_left2right_1_3
            bottom_right_corner = bottom_left_corner + bottom_left2right_1_3
        elif direction == "forward":
            top_left_corner = top_left_corner + top_left2right_1_3
            bottom_left_corner = bottom_left_corner + bottom_left2right_1_3
            top_right_corner = top_left_corner + top_left2right_1_3
            bottom_right_corner = bottom_left_corner + bottom_left2right_1_3
        elif direction == "right":
            top_left_corner = top_left_corner + top_left2right_2_3
            bottom_left_corner = bottom_left_corner + bottom_left2right_2_3

        return [list(top_left_corner), list(bottom_left_corner), list(top_right_corner), list(bottom_right_corner)]

    def project_3d_bbox_to_2d(self, sim, hint_bbox_3d: Dict) -> Union[List[int], None]:
        projected_points = []
        depths = []
        bbox_corners_3d = self.get_directed_bbox(sim, hint_bbox_3d)

        if bbox_corners_3d is None:
            return None

        for corner_3d in bbox_corners_3d:
            corner_3d = mn.Vector3(corner_3d)
            point_2d_pixels, depth = self.get_2d_point(sim, "rgb", corner_3d)
            projected_points.append(np.array(point_2d_pixels))
            depths.append(depth)

        projected_points = np.array(projected_points)
        depths = np.array(depths)
        height, width = self.config.RGB_HEIGHT, self.config.RGB_WIDTH

        x_min, y_min = np.min(projected_points, axis=0)
        x_max, y_max = np.max(projected_points, axis=0)
        if 0 <= x_min and x_max < width and 0 <= y_min and y_max < height and np.all(depths >= 0.0) and np.all(depths < 1.0):
            return [x_min, y_min, x_max, y_max]
        else:
            return None

    def get_two_closest_hints(self, hints_center_position: List[List[float]], agent_position: List[float]):
        if not hints_center_position or len(hints_center_position) < 2:
            return [0] if hints_center_position else None

        closest1 = (float('inf'), -1)
        closest2 = (float('inf'), -1)

        for i, point in enumerate(hints_center_position):
            dist_sq = (point[0] - agent_position[0])**2 + (point[2] - agent_position[2])**2
            
            if dist_sq < closest1[0]:
                closest2 = closest1
                closest1 = (dist_sq, i)
            elif dist_sq < closest2[0]:
                closest2 = (dist_sq, i)
                
        dist1 = math.sqrt(closest1[0])
        idx1 = closest1[1]
        
        dist2 = math.sqrt(closest2[0])
        idx2 = closest2[1]

        return idx1

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        hints_bbox_3d = episode.hints_bbox_3d # [{"left": []}, {"forward": []}, {"right": []}, ...]
        hints_center_position = [list(hint_bbox_3d.values())[0][0] for hint_bbox_3d in hints_bbox_3d]
        hints_center_position_xz = np.array([[hint_center_position[0], hint_center_position[2]] for hint_center_position in hints_center_position])
        agent_position_xz = np.array([agent_position[0], agent_position[2]])
        kd_tree = KDTree(hints_center_position_xz)
        _, hint_index = kd_tree.query(agent_position_xz)
        
        bbox_2d = None
        for index in range(max(0, hint_index - 1), min(len(hints_bbox_3d), hint_index + 2)):
            hint_bbox_3d = hints_bbox_3d[index]
            bbox_2d = self.project_3d_bbox_to_2d(self._sim, hint_bbox_3d)

            if bbox_2d is None:
                continue
            else: 
                if bbox_2d[2] - bbox_2d[0] < 28 or bbox_2d[3] - bbox_2d[1] < 28:
                    bbox_2d = None
                    continue
                else: 
                    direction = list(hint_bbox_3d.keys())[0]
                    if (self.enable_forward_hint_dropout and direction == "forward" and np.random.random() < self.forward_hint_dropout_prob):
                        bbox_2d = None
                    if self.enable_close_non_forward_hint_dropout and direction != "forward":
                        bbox_3d = list(hint_bbox_3d.values())[0]
                        hint_center = bbox_3d[0]
                        hint_distance = np.linalg.norm(np.array([agent_position[0], agent_position[2]]) - np.array([hint_center[0], hint_center[2]]))
                        if hint_distance < self.close_non_forward_hint_distance:
                            bbox_2d = None
                    break
        
        self.hint_position = bbox_2d

        if bbox_2d is None:
            bbox_2d = [0, 0, 0, 0]
        return np.array(bbox_2d, dtype=np.int16) 


@registry.register_sensor
class HintPosition3DSensor(Sensor):
    cls_uuid: str = "hint_position_3d"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self.hint_position = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return spaces.Box(low=0, high=max(self.config.RGB_HEIGHT, self.config.RGB_WIDTH), shape=(4,), dtype=np.uint16)

    # project a 3D point into 2D image space for a particular sensor
    def get_2d_point(self, sim, sensor_name, point_3d: mn.Vector3):
        # get the scene render camera and sensor object
        visual_sensor = sim._sensors[sensor_name]
        scene_graph = sim.get_active_scene_graph()
        scene_graph.set_default_render_camera_parameters(visual_sensor._sensor_object)
        render_camera = scene_graph.get_default_render_camera()
        # use the camera and projection matrices to transform the point onto the near plane
        projected_point_3d = render_camera.projection_matrix.transform_point(render_camera.camera_matrix.transform_point(point_3d))
        # convert the 3D near plane point to integer pixel space
        point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
        point_2d = point_2d / render_camera.projection_size()[0]
        point_2d += mn.Vector2(0.5)
        point_2d *= render_camera.viewport
        point_depth = projected_point_3d[2]
        return mn.Vector2i(point_2d), point_depth

    def get_directed_bbox(self, sim, hint_bbox_3d: Dict):
        agent_position = sim.get_agent_state().position
        agent_position[1] = 0.0 

        (direction, bbox_3d), = hint_bbox_3d.items()
        bbox_center_3d = bbox_3d[0]
        bbox_center_3d[1] = 0.0
        bbox_corners_3d = bbox_3d[1: ]

        agent_center_vec = np.array(bbox_center_3d) - np.array(agent_position)
        agent_center_vec_norm = agent_center_vec / np.linalg.norm(agent_center_vec)

        left_corners, right_corners = [], []
        for bbox_corner_3d in bbox_corners_3d:
            bbox_corner_3d_temp = [bbox_corner_3d[0], 0.0, bbox_corner_3d[2]]
            agent_corner_vec = np.array(bbox_corner_3d_temp) - np.array(agent_position)
            agent_corner_vec_norm = agent_corner_vec / np.linalg.norm(agent_corner_vec)

            cross_product = np.cross(agent_center_vec_norm, agent_corner_vec_norm)
            
            if cross_product[1] > 1e-6: # left
                left_corners.append(bbox_corner_3d)
            elif cross_product[1] < -1e-6: # right
                right_corners.append(bbox_corner_3d)

        if len(left_corners) < 2 or len(right_corners) < 2:
            return None
        
        top_left_index = np.argmax(np.array(left_corners)[:, 1]) # 1 means the y axis
        top_left_corner = left_corners[top_left_index]
        bottom_left_index = np.argmin(np.array(left_corners)[:, 1]) # 1 means the y axis
        bottom_left_corner = left_corners[bottom_left_index]
        
        top_right_index = np.argmax(np.array(right_corners)[:, 1]) # 1 means the y axis
        top_right_corner = right_corners[top_right_index]
        bottom_right_index = np.argmin(np.array(right_corners)[:, 1]) # 1 means the y axis
        bottom_right_corner = right_corners[bottom_right_index]

        top_left2right_vec = np.array(top_right_corner) - np.array(top_left_corner)
        top_left2right_1_3 = top_left2right_vec / 3
        top_left2right_2_3 = top_left2right_vec * 2 / 3

        bottom_left2right_vec = np.array(bottom_right_corner) - np.array(bottom_left_corner)
        bottom_left2right_1_3 = bottom_left2right_vec / 3
        bottom_left2right_2_3 = bottom_left2right_vec * 2 / 3

        if direction == "left":
            top_right_corner = top_left_corner + top_left2right_1_3
            bottom_right_corner = bottom_left_corner + bottom_left2right_1_3
        elif direction == "forward":
            top_left_corner = top_left_corner + top_left2right_1_3
            bottom_left_corner = bottom_left_corner + bottom_left2right_1_3
            top_right_corner = top_left_corner + top_left2right_1_3
            bottom_right_corner = bottom_left_corner + bottom_left2right_1_3
        elif direction == "right":
            top_left_corner = top_left_corner + top_left2right_2_3
            bottom_left_corner = bottom_left_corner + bottom_left2right_2_3

        return [list(top_left_corner), list(bottom_left_corner), list(top_right_corner), list(bottom_right_corner)]

    def project_3d_bbox_to_2d(self, sim, hint_bbox_3d: Dict) -> Union[List[int], None]:
        projected_points = []
        depths = []
        bbox_corners_3d = self.get_directed_bbox(sim, hint_bbox_3d)

        if bbox_corners_3d is None:
            return None

        for corner_3d in bbox_corners_3d:
            corner_3d = mn.Vector3(corner_3d)
            point_2d_pixels, depth = self.get_2d_point(sim, "rgb", corner_3d)
            projected_points.append(np.array(point_2d_pixels))
            depths.append(depth)

        projected_points = np.array(projected_points)
        depths = np.array(depths)
        height, width = self.config.RGB_HEIGHT, self.config.RGB_WIDTH

        x_min, y_min = np.min(projected_points, axis=0)
        x_max, y_max = np.max(projected_points, axis=0)
        if 0 <= x_min and x_max < width and 0 <= y_min and y_max < height and np.all(depths >= 0.0) and np.all(depths < 1.0):
            return [x_min, y_min, x_max, y_max]
        else:
            return None

    def get_two_closest_hints(self, hints_center_position: List[List[float]], agent_position: List[float]):
        if not hints_center_position or len(hints_center_position) < 2:
            return [0] if hints_center_position else None

        closest1 = (float('inf'), -1)
        closest2 = (float('inf'), -1)

        for i, point in enumerate(hints_center_position):
            dist_sq = (point[0] - agent_position[0])**2 + (point[2] - agent_position[2])**2
            
            if dist_sq < closest1[0]:
                closest2 = closest1
                closest1 = (dist_sq, i)
            elif dist_sq < closest2[0]:
                closest2 = (dist_sq, i)
                
        dist1 = math.sqrt(closest1[0])
        idx1 = closest1[1]
        
        dist2 = math.sqrt(closest2[0])
        idx2 = closest2[1]

        return idx1

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        hints_bbox_3d = episode.hints_bbox_3d # [{"left": []}, {"forward": []}, {"right": []}, ...]
        hints_center_position = [list(hint_bbox_3d.values())[0][0] for hint_bbox_3d in hints_bbox_3d]
        hints_center_position_xz = np.array([[hint_center_position[0], hint_center_position[2]] for hint_center_position in hints_center_position])
        agent_position_xz = np.array([agent_position[0], agent_position[2]])
        kd_tree = KDTree(hints_center_position_xz)
        _, hint_index = kd_tree.query(agent_position_xz)
        
        bbox_2d = None
        for index in range(max(0, hint_index - 1), min(len(hints_bbox_3d), hint_index + 2)):
            hint_bbox_3d = hints_bbox_3d[index]
            bbox_2d = self.project_3d_bbox_to_2d(self._sim, hint_bbox_3d)

            if bbox_2d is None:
                continue
            else: 
                if bbox_2d[2] - bbox_2d[0] < 28 or bbox_2d[3] - bbox_2d[1] < 28:
                    bbox_2d = None
                    continue
                else: 
                    break
        
        if bbox_2d is None:
            hint_bbox_3d = {}
        return hint_bbox_3d
    

@registry.register_sensor
class HintSensor(HintPositionSensor):
    cls_uuid: str = "hint"

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        self.hint_image_shape = (self.config.HEIGHT, self.config.WIDTH, self.config.CHANNEL)
        return spaces.Box(low=0, high=255, shape=self.hint_image_shape, dtype=np.uint8)

    def crop_image_from_bbox(self, full_image: np.ndarray, bbox_2d: List[int]) -> Union[np.ndarray, None]:
        x_min, y_min, x_max, y_max = bbox_2d
        cropped_region = full_image[y_min:y_max, x_min:x_max]
        
        return cropped_region

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        observations = self._sim.get_sensor_observations()
        rgb_image = observations["rgb"]

        super().get_observation(episode=episode)
        bbox_2d = self.hint_position

        if bbox_2d is None:
            hint_image = np.zeros(self.hint_image_shape, dtype=np.uint8)
        else:
            cropped_region = self.crop_image_from_bbox(rgb_image, bbox_2d)
            hint_image = cropped_region[:, :, :self.config.CHANNEL] # remove alpha channel, [H, W, C]
            # Note: OpenCV's dsize is in (width, height) format
            hint_image = cv2.resize(hint_image, (self.config.WIDTH, self.config.HEIGHT), interpolation=cv2.INTER_LINEAR)

        return hint_image


@registry.register_sensor
class HintLanguageSensor(HintPositionSensor):
    cls_uuid: str = "hint_language"

    direction_to_token = {
        "none": 0,
        "forward": 1,
        "left": 2,
        "right": 3,
    }

    token_to_description = {
        0: "",
        1: "go forward",
        2: "turn left",
        3: "turn right",
    }

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return spaces.Box(low=0, high=3, shape=(1,), dtype=np.int64)

    def get_visible_hint_bbox_3d(self, episode) -> Dict:
        agent_position = self._sim.get_agent_state().position
        hints_bbox_3d = episode.hints_bbox_3d

        if not hints_bbox_3d:
            return {}

        hints_center_position = [list(hint_bbox_3d.values())[0][0] for hint_bbox_3d in hints_bbox_3d]
        hints_center_position_xz = np.array([[hint_center_position[0], hint_center_position[2]] for hint_center_position in hints_center_position])
        agent_position_xz = np.array([agent_position[0], agent_position[2]])
        kd_tree = KDTree(hints_center_position_xz)
        _, hint_index = kd_tree.query(agent_position_xz)

        for index in range(max(0, hint_index - 1), min(len(hints_bbox_3d), hint_index + 2)):
            hint_bbox_3d = hints_bbox_3d[index]
            bbox_2d = self.project_3d_bbox_to_2d(self._sim, hint_bbox_3d)

            if bbox_2d is None:
                continue

            if bbox_2d[2] - bbox_2d[0] < 28 or bbox_2d[3] - bbox_2d[1] < 28:
                continue

            direction = list(hint_bbox_3d.keys())[0]
            if (self.enable_forward_hint_dropout and direction == "forward" and np.random.random() < self.forward_hint_dropout_prob):
                return {}

            return hint_bbox_3d

        return {}

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        hint_bbox_3d = self.get_visible_hint_bbox_3d(episode)
        if not hint_bbox_3d:
            return np.array([self.direction_to_token["none"]], dtype=np.int64)

        direction = list(hint_bbox_3d.keys())[0]
        token = self.direction_to_token.get(direction, self.direction_to_token["none"])
        return np.array([token], dtype=np.int64)
