import gzip
import json

import ipdb
import torch
import tqdm
import random
import lmdb
import numpy as np
import msgpack_numpy
import torch.distributed as dist
from collections import defaultdict, deque

from gym import Space
from habitat.config.default import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import apply_obs_transforms_obs_space, get_active_obs_transforms

from pwgw_baselines.common.env_utils import construct_envs
from pwgw_baselines.common.utils import extract_instruction_tokens


STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """
    Each sample in batch:
    (obs, selected_actions, oracle_actions, inflec_weight)
    one batch is a list:
    [sample_1, sample_2, sample_3, ..., sample_batch_size]
    """


    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations     = list(transposed[0]) # [observations_env1, observations_env2, ...]
    selected_actions = list(transposed[1])
    oracle_actions   = list(transposed[2])
    weights          = list(transposed[3])

    new_observations_batch = defaultdict(list)
    B = len(selected_actions) 
    for sensor in observations[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(observations[bid][sensor])
    observations = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in selected_actions) 
    for bid in range(B):
        for sensor in observations:
            observations[sensor][bid] = _pad_helper(observations[sensor][bid], max_traj_len, fill_val=1.0)

        selected_actions[bid] = _pad_helper(selected_actions[bid], max_traj_len)
        oracle_actions[bid]   = _pad_helper(oracle_actions[bid], max_traj_len)
        weights[bid]          = _pad_helper(weights[bid], max_traj_len)

    for sensor in observations:
        observations[sensor] = torch.stack(observations[sensor], dim=1) # [L, B, H, W, C]
    observations = ObservationsDict(observations)

    selected_actions = torch.stack(selected_actions, dim=1) # [63, 3]
    oracle_actions   = torch.stack(oracle_actions, dim=1) # [63, 3]
    weights          = torch.stack(weights, dim=1) # [63, 3]

    return (observations, selected_actions, oracle_actions, weights)


class TeacherRecollectDataset(torch.utils.data.IterableDataset):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._preload = deque()

        assert (config.IL.RECOLLECT_TRAINER.preload_size >= config.IL.batch_size), "preload size must be greater than batch size."
        
        self.envs = None
        self._env_observations = None

        if config.IL.use_iw:
            self.inflec_weights = torch.tensor([1.0, config.IL.inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        if self.config.IL.RECOLLECT_TRAINER.preload_trajectories_file:
            with gzip.open(config.IL.RECOLLECT_TRAINER.trajectories_file, "rt") as f:
                self.trajectories = json.load(f)
        else:
            self.trajectories = self.collect_dataset()

        self.initialize_sims()

    @property
    def batch_size(self):
        return self.config.IL.batch_size

    @property
    def observation_space(self) -> Space:
        assert self.envs is not None, "Simulator must first be loaded."
        assert self._observation_space is not None
        return self._observation_space

    @property
    def action_space(self) -> Space:
        assert self.envs is not None, "Simulator must first be loaded."
        return self.envs.action_spaces[0]

    def close_sims(self):
        self.envs.close()
        del self.envs
        del self._env_observations
        self.envs = None
        self._env_observations = None

    def collect_dataset(self):
        """
        Uses the ground truth trajectories to create a teacher forcing datset for a given split.
        Loads both guide and follower episodes.
        """
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT

        with gzip.open(self.config.IL.RECOLLECT_TRAINER.gt_path.format(split=split)) as f:
            gt_data = json.load(f)

        t = (tqdm.tqdm(gt_data.items(), "GT Collection") if self.config.use_pbar else gt_data.items())

        for episode_id, trajectory in t:
            if (self.config.IL.RECOLLECT_TRAINER.max_traj_len != -1 and
                len(trajectory["actions"]) > self.config.IL.RECOLLECT_TRAINER.max_traj_len):
                
                continue

            for i, action in enumerate(trajectory["actions"]):
                prev_action = (trajectories[episode_id][i - 1][1] if i else HabitatSimActions.STOP)

                # [prev_action, action, oracle_action]
                trajectories[episode_id].append([prev_action, action, action])

        with gzip.open(self.config.IL.RECOLLECT_TRAINER.trajectories_file, "wt") as f:
            f.write(json.dumps(trajectories))

        return trajectories

    def initialize_sims(self):
        config = self.config.clone()
        config.defrost()
        config.TASK_CONFIG.MEASUREMENTS = [] 
        config.freeze()

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME), episodes_allowed=list(self.trajectories.keys()))
        self.length = sum(self.envs.number_of_episodes)

        self.obs_transforms = get_active_obs_transforms(self.config)
        self._observation_space = apply_obs_transforms_obs_space(self.envs.observation_spaces[0], self.obs_transforms)

        self.env_step = [0 for _ in range(self.envs.num_envs)]
        self._env_observations = [[] for _ in range(self.envs.num_envs)]

        observations = self.envs.reset()
        # just remove the instruction text and remain the instruction token
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)

        # put current trajectory action into the envs
        for i, ep in enumerate(self.envs.current_episodes()):
            path_step = self.trajectories[ep.episode_id][0] # the first (current) trajectory action: [prev_action, action, oracle_action]
            self._env_observations[i].append(
                (
                    observations[i],
                    path_step[1], # selected_action
                    path_step[2], # oracle_action
                )
            )

    def _load_next(self):
        """
        Episode length is currently not considered. We were previously batching episodes
        together with similar lengths. Not sure if we need to bring that back.
        """

        if len(self._preload):
            print("#############################", len(self._preload), "##############################")
            return self._preload.popleft()

        while (len(self._preload) < self.config.IL.RECOLLECT_TRAINER.preload_size):
            current_episodes = self.envs.current_episodes()
            prev_eps = current_episodes

            actions = [self.trajectories[ep.episode_id][self.env_step[i]][1] for i, ep in enumerate(current_episodes)]
            outputs = self.envs.step(actions)
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)

            current_episodes = self.envs.current_episodes()

            for i in range(self.envs.num_envs):
                self.env_step[i] += 1
                
                if dones[i]:
                    assert len(self._env_observations[i]) == len(self.trajectories[prev_eps[i].episode_id]),\
                        "Collected episode does not match the step count of trajectory"
                    
                    self._preload.append(
                        (
                            [o[0] for o in self._env_observations[i]], 
                            [o[1] for o in self._env_observations[i]], 
                            [o[2] for o in self._env_observations[i]], 
                        )
                    )

                    self._env_observations[i] = []
                    self.env_step[i] = 0

                path_step = self.trajectories[current_episodes[i].episode_id][self.env_step[i]]
                
                self._env_observations[i].append((observations[i], path_step[1], path_step[2]))

                assert (len(self._env_observations[i]) <= self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS),\
                    "Trajectories should be no more than the maximum episode steps."

        return self._preload.popleft()

    def __next__(self):
        """
        Takes about 1s to once self._load_next() has finished with a batch size of 5.
        For this reason, we probably don't need to use extra workers.
        """

        x = self._load_next()
        obs, selected_actions, oracle_actions = x 

        # transpose the observations
        obs_t = defaultdict(list)
        for k in obs[0]:
            for i in range(len(obs)):
                obs_t[k].append(obs[i][k])

            obs_t[k] = np.array(obs_t[k])

        for k, v in obs_t.items():
            obs_t[k] = torch.from_numpy(np.copy(v))

        # transpose the selected_actions
        selected_actions = torch.from_numpy(np.copy(selected_actions))
        # transpose the oracle_actions
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))
        # inflection weights
        inflections = torch.cat([torch.tensor([1], dtype=torch.long), (oracle_actions[1:] != oracle_actions[:-1]).long()])

        return (obs_t, selected_actions, oracle_actions, self.inflec_weights[inflections])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert (worker_info.num_workers == 1), "multiple workers not supported."

        return self


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(self, lmdb_features_dir, use_iw, inflection_weight_coef=1.0, lmdb_map_size=1e9, batch_size=1):
        super().__init__()

        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = 10 
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(self.lmdb_features_dir, map_size=int(self.lmdb_map_size), readonly=True, lock=False) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank_length = int(np.floor(self.length / world_size))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(self.lmdb_features_dir, map_size=int(self.lmdb_map_size), readonly=True, lock=False) as lmdb_env,\
                lmdb_env.begin(buffers=True) as txn:
                
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    new_preload.append(msgpack_numpy.unpackb(txn.get(str(self.load_ordering.pop()).encode()), raw=False))
                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs, selected_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        selected_actions = torch.from_numpy(np.copy(selected_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))
        inflections = torch.cat([torch.tensor([1], dtype=torch.long), (oracle_actions[1:] != oracle_actions[:-1]).long()])

        return (obs, selected_actions, oracle_actions, self.inflec_weights[inflections])

    def __iter__(self):
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # per_worker = int(np.ceil(self.length / (num_workers * world_size)))
        per_worker = int(np.floor(self.length / (num_workers * world_size)))
        start = per_worker * (worker_id + rank * num_workers)
        end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(reversed(_block_shuffle(list(range(start, end)), self.preload_size)))
        # print("self.load_ordering:", self.load_ordering)

        return self