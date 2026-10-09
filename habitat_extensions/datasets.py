import os
import gzip
import json
from typing import Dict, List, Optional, Union

import attr
from habitat.config import Config
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.utils import VocabDict
from habitat.tasks.vln.vln import InstructionData, VLNEpisode
from habitat.tasks.nav.nav import NavigationGoal, NavigationEpisode, NavigationTask


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_EPISODES_MASK = "*"


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = attr.ib(default=None)
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(default=None, validator=not_none_validator)
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


@registry.register_dataset(name="VLN-CE-v1")
class VLNCEDatasetV1(Dataset):
    """Loads the R2R VLN-CE dataset"""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                e
                for e in self.episodes
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]

        if ALL_EPISODES_MASK not in config.EPISODES_ALLOWED:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.EPISODES_ALLOWED)
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:

        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(word_list=deserialized["instruction_vocab"]["word_list"])

        for episode in deserialized["episodes"]:
            # cast integer IDs to strings
            episode["episode_id"] = str(episode["episode_id"])
            episode["trajectory_id"] = str(episode["trajectory_id"])

            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted({cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes})

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(config.DATA_PATH.format(split=config.SPLIT)) and os.path.exists(config.SCENES_DIR)


@attr.s(auto_attribs=True, kw_only=True)
class PWGWEpisode(NavigationEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)
    hints_bbox_3d: Optional[List[Dict]] = attr.ib(default=None)
    object_category: Optional[str] = attr.ib(default=None)


@registry.register_dataset(name="PWGW-CE")
class PWGWCEDataset(Dataset):
    """Loads the PWGW-CE dataset"""

    episodes: List[VLNEpisode]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                e
                for e in self.episodes
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]

        if ALL_EPISODES_MASK not in config.EPISODES_ALLOWED:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.EPISODES_ALLOWED)
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:

        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            # cast integer IDs to strings
            episode["episode_id"] = str(episode["episode_id"])
            episode["trajectory_id"] = str(episode["trajectory_id"])

            episode = PWGWEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted({cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes})

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(config.DATA_PATH.format(split=config.SPLIT)) and os.path.exists(config.SCENES_DIR)


@registry.register_task(name="PWGW-Nav")
class PWGWTask(NavigationTask):
    r"""Point Where Go Where Navigation Task
    Goal: An agent must navigate to a goal location in a 3D environment
        specified by a natural language instruction.
    Metric: Success weighted by Path Length (SPL)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
