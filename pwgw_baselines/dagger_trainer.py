import gc
import os
import cv2
import copy
import lmdb
import tqdm
import torch
import warnings
import numpy as np
import msgpack_numpy
import torch.distributed as dist
import torchvision.transforms.functional as TF

from collections import deque, defaultdict

from habitat import logger
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

from pwgw_baselines.common.aux_losses import AuxLosses
from pwgw_baselines.common.base_pwgw_trainer import BasePWGWTrainer, process_observations
from pwgw_baselines.common.env_utils import construct_envs, construct_envs_auto_reset_false
from pwgw_baselines.common.datasets import collate_fn, IWTrajectoryDataset
from pwgw_baselines.common.utils import process_observations

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf


def observation_logs(observation, step_id):
    rgb_image = observation["rgb"]
    rgb_image = rgb_image[..., 0:3][..., ::-1] # change first to RGB and then change to BGR
    rgb_image = np.ascontiguousarray(rgb_image)

    hint_position = observation["hint_position"]
    top_left_corner = (hint_position[0], hint_position[1])
    bottom_right_corner = (hint_position[2], hint_position[3])
    cv2.rectangle(rgb_image, top_left_corner, bottom_right_corner, (0, 255, 0), 2)
    
    hint_image = observation["hint"]
    hint_image = hint_image[..., 0:3][..., ::-1] # change first to RGB and then change to BGR

    cv2.imwrite(f"/home/nusssun/NAS/Jasun.d/point_where_go_where/data/image_logs/rgb_image_{step_id}.png", rgb_image)


@baseline_registry.register_trainer(name="dagger_trainer")
class DaggerTrainer(BasePWGWTrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(split=config.TASK_CONFIG.DATASET.SPLIT)
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def _update_dataset(self, data_it):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.cpu())
            return hook

        rgb_features = None
        rgb_hook = None
        if not self.config.MODEL.RGB_ENCODER.trainable:
            rgb_features = torch.zeros((1, ), device="cpu")
            rgb_hook = self.policy.module.net.rgb_encoder.register_forward_hook(hook_builder(rgb_features))

        depth_features = None
        depth_hook = None
        if not self.config.MODEL.DEPTH_ENCODER.trainable:
            depth_features = torch.zeros((1, ), device="cpu")
            depth_hook = self.policy.module.net.depth_encoder.visual_encoder.register_forward_hook(hook_builder(depth_features))

        hint_features = None
        hint_hook = None
        if not self.config.MODEL.HINT_ENCODER.trainable:
            hint_features = torch.zeros((1, ), device="cpu")
            hint_hook = self.policy.module.net.hint_encoder.hint_image_encoder.register_forward_hook(hook_builder(hint_features))

        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid
        success_distance = self.config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE

        envs = construct_envs_auto_reset_false(self.config, get_env_class(self.config.ENV_NAME))
        observations = envs.reset() 

        processed_observations = process_observations(copy.deepcopy(observations))
        batch = batch_obs(processed_observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        step_id   = -1 
        endeds    = np.array([False for _ in range(envs.num_envs)]) 
        selected_actions = torch.zeros(envs.num_envs, 1, dtype=torch.long, device=self.device)
        hist_embeds = deque(maxlen=self.config.MODEL.max_hist_len)
        hist_lens = [hist_embeds.maxlen for _ in range(envs.num_envs)]
        episodes  = [[] for _ in range(envs.num_envs)]
        dones     = [False for _ in range(envs.num_envs)] 
        invalid_episodes = np.array([False for _ in range(envs.num_envs)]) 

        # initialization
        hist_embed = self.policy.module.net(mode="history", observations=batch) # [B, 768]
        while len(hist_embeds) < hist_embeds.maxlen:
            hist_embeds.append(hist_embed)

        p = self.config.IL.DAGGER.p 
        beta = 0.0 if p == 0.0 else p ** data_it 
        ensure_unique_episodes = beta == 1.0

        collected_eps = 0
        ep_ids_collected = None
        if ensure_unique_episodes:
            ep_ids_collected = {ep.episode_id for ep in envs.current_episodes()}

        with tqdm.tqdm(total=self.config.IL.DAGGER.update_size, dynamic_ncols=True) as pbar,\
        lmdb.open(self.lmdb_features_dir, map_size=int(self.config.IL.DAGGER.lmdb_map_size)) as lmdb_env, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)

            while collected_eps < self.config.IL.DAGGER.update_size:
                if endeds.all():
                    for i in range(envs.num_envs):
                        if not invalid_episodes[i]:
                            ep = episodes[i]

                            if len(ep) == self.config.MODEL.max_action_steps: continue
                            
                            traj_obs = batch_obs([step[0] for step in ep], device=torch.device("cpu"))
                            del traj_obs[expert_uuid] 

                            for k, v in traj_obs.items():
                                traj_obs[k] = v.numpy()
                                if self.config.IL.DAGGER.lmdb_fp16:
                                    traj_obs[k] = traj_obs[k].astype(np.float16)

                            transposed_ep = [traj_obs, np.array([step[1] for step in ep], dtype=np.int64), np.array([step[2] for step in ep], dtype=np.int64)]
                            txn.put(str(start_id + collected_eps).encode(), msgpack_numpy.packb(transposed_ep, use_bin_type=True))

                            pbar.update()
                            collected_eps += 1

                            if (collected_eps % self.config.IL.DAGGER.lmdb_commit_frequency) == 0:
                                txn.commit()
                                txn = lmdb_env.begin(write=True)

                            del ep, traj_obs, transposed_ep

                    observations = envs.reset()
                    if ensure_unique_episodes:
                        envs_to_pause = []
                        current_episodes = envs.current_episodes()

                        for i in range(envs.num_envs):
                            if (current_episodes[i].episode_id in ep_ids_collected):
                                envs_to_pause.append(i)
                            else:
                                ep_ids_collected.add(current_episodes[i].episode_id)

                        (envs, observations) = self._pause_envs(envs_to_pause, envs, observations)
                        if envs.num_envs == 0: break

                    processed_observations = process_observations(copy.deepcopy(observations))
                    batch = batch_obs(processed_observations, self.device)
                    batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                    hist_embeds.clear()
                    hist_embed = self.policy.module.net(mode="history", observations=batch) # [B, 768]
                    while len(hist_embeds) < hist_embeds.maxlen:
                        hist_embeds.append(hist_embed)

                    step_id          = -1
                    endeds           = np.array([False for _ in range(envs.num_envs)])
                    selected_actions = torch.zeros(envs.num_envs, 1, dtype=torch.long, device=self.device)
                    hist_lens        = [hist_embeds.maxlen for _ in range(envs.num_envs)]
                    episodes         = [[] for _ in range(envs.num_envs)]
                    invalid_episodes = np.array([False for _ in range(envs.num_envs)])

                # make prediction using current observation
                model_inputs = {"mode":          "observation",
                                "observations":  batch,
                                "hist_embeds":   list(hist_embeds),
                                "hist_lens":     hist_lens,
                                "deterministic": False,
                                }
                visual_embed_spatial, actions = self.policy.module.act(**model_inputs)

                actions = torch.where(torch.rand_like(actions, dtype=torch.float) < beta, batch[expert_uuid].long(), actions)

                skips = batch[expert_uuid].long() == -1 
                actions = torch.where(skips, torch.zeros_like(actions), actions)
                selected_actions.copy_(actions)

                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True).numpy()
                invalid_episodes = np.logical_or(invalid_episodes, skips)

                for i in range(envs.num_envs):
                    if rgb_features is not None:
                        processed_observations[i]["rgb_features"] = rgb_features[i]
                        del processed_observations[i]["rgb"]

                    if depth_features is not None:
                        processed_observations[i]["depth_features"] = depth_features[i]
                        del processed_observations[i]["depth"]

                    if hint_features is not None:
                        processed_observations[i]["hint_features"] = hint_features[i]
                        del processed_observations[i]["hint"]

                    if not endeds[i]: 
                        episodes[i].append((processed_observations[i], selected_actions[i].item(), batch[expert_uuid][i].item()))

                # after the prediction, we can translate the current observation into history info
                step_id += 1 
                model_inputs = {"mode":            "history",
                                "observations":    visual_embed_spatial,
                                "selected_action": selected_actions,
                                }

                hist_embed = self.policy.module.net(**model_inputs)
                hist_embeds.append(hist_embed)

                if step_id != 0: 
                    for i in range(envs.num_envs):
                        distance_to_goal = infos[i]["distance_to_goal"]
                        if distance_to_goal <= success_distance:
                            actions[i] = 0 

                outputs = envs.step([a[0].item() for a in actions])
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                endeds[:] = np.logical_or(endeds, np.array(dones))

                processed_observations = process_observations(copy.deepcopy(observations))
                batch = batch_obs(processed_observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            txn.commit()

        envs.close()
        envs = None

        if rgb_hook is not None:
            rgb_hook.remove()

        if depth_hook is not None:
            depth_hook.remove()

        if hint_hook is not None:
            hint_hook.remove()

    # Main method for training DAgger
    def train(self) -> None:
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True)
            except lmdb.Error as err:
                logger.error("Cannot open database for teacher forcing preload.")
                raise err
        else: 
            with lmdb.open(self.lmdb_features_dir, map_size=int(self.config.IL.DAGGER.lmdb_map_size)) as lmdb_env,\
            lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        # if doing teacher forcing, don't switch the scene until it is completed
        self.config.defrost()
        if self.config.IL.DAGGER.p == 1.0: 
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (-1)
        self.config.freeze()

        observation_space, action_space = self._get_spaces(self.config)
        self._initialize_policy(self.config, self.config.IL.load_from_ckpt, observation_space=observation_space, action_space=action_space)

        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs, purge_step=0) as writer:
            for dagger_it in range(self.config.IL.DAGGER.iterations):
                batch_id = 0
                # if not self.config.IL.DAGGER.preload_lmdb_features:
                if not self.config.IL.DAGGER.preload_lmdb_features and self.local_rank == 0:
                    # dataset aggregate for DAGGER training
                    self._update_dataset(dagger_it + (1 if self.config.IL.load_from_ckpt else 0))

                dist.barrier()

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()

                dataset = IWTrajectoryDataset(self.lmdb_features_dir,
                                              self.config.IL.use_iw,
                                              inflection_weight_coef=self.config.IL.inflection_weight_coef,
                                              lmdb_map_size=self.config.IL.DAGGER.lmdb_map_size,
                                              batch_size=self.config.IL.batch_size)

                diter = torch.utils.data.DataLoader(dataset,
                                                    batch_size=self.config.IL.batch_size, 
                                                    shuffle=False, 
                                                    # sampler=dataset_sampler,
                                                    collate_fn=collate_fn,
                                                    pin_memory=False,
                                                    drop_last=True,  # drop last batch if smaller
                                                    num_workers=2, 
                                                    )

                AuxLosses.activate()
                for epoch in tqdm.trange(self.config.IL.epochs, dynamic_ncols=True, disable=(self.local_rank != 0)):
                    average_loss = []
                    for batch in tqdm.tqdm(diter, total=dataset.rank_length // dataset.batch_size, leave=False, dynamic_ncols=True, disable=(self.local_rank != 0)):
                        (observations, selected_actions, oracle_actions, weights) = batch

                        observations     = {k: v.to(device=self.device, dtype=torch.float32, non_blocking=True) for k, v in observations.items()}
                        selected_actions = selected_actions.to(device=self.device, non_blocking=True)
                        oracle_actions   = oracle_actions.to(device=self.device, non_blocking=True)
                        weights          = weights.to(device=self.device, non_blocking=True)
                
                        loss, action_loss, aux_loss = self._update_agent(observations, selected_actions, oracle_actions, weights)

                        # aggregate the loss across all GPUs
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM) # sum all local losses
                        loss /= dist.get_world_size() # average the total loss
                        average_loss.append(loss.item())

                        if self.local_rank == 0:
                            writer.add_scalar(f"train_loss_iter_{dagger_it}", loss.item(), batch_id)

                        batch_id += 1

                    if self.local_rank == 0:
                        self.save_checkpoint(f"ckpt_{dagger_it}_{epoch}.pth")

                AuxLosses.deactivate()
