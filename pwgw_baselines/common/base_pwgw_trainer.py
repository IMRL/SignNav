import os
import cv2
import copy
import time
import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datetime import datetime, timedelta

from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch, apply_obs_transforms_obs_space, get_active_obs_transforms
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.common import batch_obs

from habitat_extensions import maps
from habitat_extensions.utils import generate_video, observations_to_image
from pwgw_baselines.common.aux_losses import AuxLosses
from pwgw_baselines.common.env_utils import construct_envs_auto_reset_false
from pwgw_baselines.common.utils import extract_position_yaw, process_observations, save_agent_trajectory

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class BasePWGWTrainer(BaseILTrainer):
    """A base trainer for PWGW imitation learning."""

    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # self.local_rank = self.config.TORCH_GPU_ID
        # self.device = (torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu"))

        dist.init_process_group(backend='nccl', timeout=timedelta(hours=48)) 
        self.local_rank = torch.distributed.get_rank() 
        torch.cuda.set_device(self.local_rank) 
        self.device = torch.device(f"cuda:{self.local_rank}")

    def _initialize_policy(self, config: Config, load_from_ckpt: bool, observation_space: Space, action_space: Space) -> None:
        # get the definition of "policy_name" class
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        # initialize an policy instance
        self.policy = policy.from_config(config=config, observation_space=observation_space, action_space=action_space)
        # now we have the policy network model!
        self.policy.to(self.device)
        self.policy = torch.nn.parallel.DistributedDataParallel(self.policy, device_ids=[self.local_rank], find_unused_parameters=True, output_device=self.local_rank)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.IL.lr)
        # self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location={"cuda:0": f"cuda:{self.local_rank}"})
            self.policy.load_state_dict(ckpt_dict["state_dict"])

            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    def save_checkpoint(self, file_name: str) -> None:
        save_path = os.path.join(self.config.CHECKPOINT_FOLDER, self.timestamp)
        os.makedirs(save_path, exist_ok=True)

        checkpoint = {"state_dict": self.policy.state_dict(), "config": self.config}
        torch.save(checkpoint, os.path.join(save_path, file_name))

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def _get_spaces(self, config: Config, envs: Optional[Any] = None) -> Tuple[Space]:
        """
        Get both the observation space and action space.

        Args:
            config (Config): The config that specifies the observation transforms.
            envs (Any, optional): An existing Environment. If None, an environment is created using the config.

        Returns:
            observation space, action space
        """

        if envs is not None:
            observation_space = envs.observation_spaces[0]
            action_space = envs.action_spaces[0]
        else:
            env = get_env_class(self.config.ENV_NAME)(config=config)
            observation_space = env.observation_space
            action_space = env.action_space

            env.close()
            del env

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(observation_space, self.obs_transforms)

        return observation_space, action_space

    def calculate_kld_loss(self, predicted_attention_maps, trajectory_heatmaps, hist_lens, intermediate_size: int = 28):
        """
            predicted_attention_maps: [L, B, 7, 7]
            trajectory_heatmaps: [L, B, 672, 896]
            hist_lens: [B]
        """

        L, B, _, _ = predicted_attention_maps.shape

        predicted_attention_maps = predicted_attention_maps.permute(1, 0, 2, 3) # [B, L, 7, 7]
        trajectory_heatmaps = trajectory_heatmaps.permute(1, 0, 2, 3) # [B, L, 672, 896]

        # predicted_attention_maps: [B, L, 28, 28]
        predicted_attention_maps = F.interpolate(predicted_attention_maps, size=(intermediate_size, intermediate_size), mode='bilinear', align_corners=False)
        predicted_attention_maps = torch.flatten(predicted_attention_maps, 2) # [B, L, 784]
        pred_log_dist = F.log_softmax(predicted_attention_maps, dim=-1) 

        downsampler = nn.AdaptiveAvgPool2d((intermediate_size, intermediate_size))
        trajectory_heatmaps = downsampler(trajectory_heatmaps.float()) # [B, L, 28, 28]
        trajectory_heatmaps = torch.flatten(trajectory_heatmaps, 2) # [B, L, 784]
        heatmap_sums = trajectory_heatmaps.sum(dim=-1, keepdim=True)
        gt_prob_dist = trajectory_heatmaps / (heatmap_sums + 1e-8)

        kld_loss_fn = nn.KLDivLoss(reduction='none')
        loss_per_element = kld_loss_fn(pred_log_dist, gt_prob_dist) # [B, L, 784]
        loss_per_hist_step = loss_per_element.sum(dim=-1) # [B, L]

        mask = torch.arange(L, device=hist_lens.device).expand(B, L) < hist_lens.unsqueeze(1)
        masked_loss = loss_per_hist_step * mask.float() 

        final_loss = masked_loss.sum() / (mask.sum() + 1e-8)

        return final_loss

    def _update_agent(self, observations, selected_actions, oracle_actions, weights, step_grad: bool=True, loss_accumulation_scalar: int=1):
        IGNORE_ID = self.config.MODEL.ignore_id
        episode_length, batch_size = oracle_actions.size()
        if self.config.MODEL.episode_length:
            episode_length = self.config.MODEL.episode_length

        predicted_actions = []
        hist_embeds = deque(maxlen=self.config.MODEL.max_hist_len)
        hist_lens = [hist_embeds.maxlen for _ in range(batch_size)]
        endeds = np.array([False for _ in range(batch_size)])

        # observations["rgb"]: [L, B, H, W, C], selected_actions: [L, B], oracle_actions: [L, B], weights: [L, B], heatmaps: [L, B, H, W]
        for step_id in range(episode_length):
            observation         = {k: v[step_id] for k, v in observations.items()} # [B, H, W, C]
            selected_action     = selected_actions[step_id] # [B]
            selected_action_not = selected_action.logical_not() # [B]
            oracle_action_not   = oracle_actions[step_id].logical_not() # [B]

            endeds[:] = np.logical_or(endeds, selected_action_not.cpu().numpy())

            if len(hist_embeds) == 0: 
                hist_embed = self.policy.module.net(mode="history", observations=observation) # [B, 768]
                while len(hist_embeds) < hist_embeds.maxlen:
                    hist_embeds.append(hist_embed)

            # make prediction using current observation
            model_inputs = {"mode":         "observation",
                            "observations": observation,
                            "hist_embeds":  list(hist_embeds),
                            "hist_lens":    hist_lens,
                            }

            visual_embed_spatial, visual_attention_map, action_distribution = self.policy.module.build_distribution(**model_inputs)
            action_logits = action_distribution.logits # [B, num_actions]
            predicted_actions.append(action_logits)

            for i, ended in enumerate(endeds):
                if ended:
                    oracle_actions[step_id][i] = IGNORE_ID

            # after the prediction, we can translate the current observation into history info
            model_inputs = {"mode":            "history",
                            "observations":    visual_embed_spatial,
                            "selected_action": selected_action,
                            }

            hist_embed = self.policy.module.net(**model_inputs)
            hist_embeds.append(hist_embed)

        predicted_actions = torch.stack(predicted_actions, dim=0) # [L, B, num_actions]
        predicted_actions = predicted_actions.permute(0, 2, 1) # [L, num_actions, B]

        AuxLosses.clear()

        # compute action loss
        action_loss = F.cross_entropy(predicted_actions, oracle_actions[0:episode_length], ignore_index=IGNORE_ID, reduction="none")
        action_loss = ((weights[0:episode_length] * action_loss).sum(0) / weights.sum(0)).mean()
        
        # compute aux_loss
        aux_mask = (weights > 0).view(-1)
        aux_loss = AuxLosses.reduce(aux_mask)
        
        # total loss
        loss = action_loss + aux_loss
        loss = loss / loss_accumulation_scalar

        loss.backward()
        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.item()
            
        # return loss.item(), action_loss.item(), aux_loss
        return loss, action_loss, aux_loss

    @staticmethod
    def _pause_envs(envs_to_pause, envs, observations):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause): 
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            observations = [observations[i] for i in state_index]

        return (envs, observations)

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0) -> None:
        config = self.config.clone()
        if self.config.EVAL.USE_CKPT_CONFIG:
            ckpt = self.load_checkpoint(checkpoint_path, map_location="cpu")
            config = self._setup_eval_config(ckpt)

        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (-1)
        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE") # "TOP_DOWN_MAP"

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(config.RESULTS_DIR, f"{split}_result.json")
            if os.path.exists(fname):
                os.remove(fname)

        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        observation_space, action_space = self._get_spaces(config, envs=envs)
        self._initialize_policy(config, load_from_ckpt=True, observation_space=observation_space, action_space=action_space)
        self.policy.eval()

        observations = envs.reset()
        observations = extract_position_yaw(observations, config.TASK_CONFIG.TASK.GLOBAL_GPS_SENSOR_UUID)
        processed_observations = process_observations(copy.deepcopy(observations))
        batch = batch_obs(processed_observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        step_id = -1
        stats_episodes = {}
        endeds = np.array([False for _ in range(envs.num_envs)])
        selected_actions = torch.zeros(envs.num_envs, 1, dtype=torch.long, device=self.device)
        hist_embeds = deque(maxlen=config.MODEL.max_hist_len)
        hist_lens = [hist_embeds.maxlen for _ in range(envs.num_envs)]

        hist_embed = self.policy.module.net(mode="history", observations=batch) # [B, 768]
        while len(hist_embeds) < hist_embeds.maxlen:
            hist_embeds.append(hist_embed)

        if "GLOBAL_GPS_SENSOR" in config.TASK_CONFIG.TASK.SENSORS:
            agent_positions = [[] for _ in range(envs.num_envs)]
            agent_yaws      = [[] for _ in range(envs.num_envs)]

        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (f"[Ckpt: {checkpoint_index}]"
                    "[Episodes evaluated: {evaluated}/{total}]"
                    "[Time elapsed (s): {time}]")

        success_distance = config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE
        start_time = time.time()
        while len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()

            with torch.no_grad():
                model_inputs = {"mode":          "observation",
                                "observations":  batch,
                                "hist_embeds":   list(hist_embeds),
                                "hist_lens":     hist_lens,
                                "deterministic": not config.EVAL.SAMPLE,
                                }

                visual_embed_spatial, actions = self.policy.module.act(**model_inputs)
                # visual_embed_spatial, actions, attention_output = self.policy.module.act(**model_inputs) # NOTE: only for temporary test
                selected_actions.copy_(actions)

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
                        print(f"! ! ! Early STOP ! ! !")

            if "GLOBAL_GPS_SENSOR" in config.TASK_CONFIG.TASK.SENSORS:
                for i in range(envs.num_envs):
                    agent_positions[i].append(observations[i]["agent_position"])
                    agent_yaws[i].append(observations[i]["agent_yaw"])

            self.observation_logging(observations[0], step_id)
            print(f"step_id: {step_id} ---> action: {actions[0].item()}")

            outputs = envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            observations = extract_position_yaw(observations, config.TASK_CONFIG.TASK.GLOBAL_GPS_SENSOR_UUID)
            processed_observations = process_observations(copy.deepcopy(observations))
            batch = batch_obs(processed_observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            for i in range(envs.num_envs):
                if endeds[i]: continue

                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(frame, current_episodes[i].instruction.instruction_text)
                    rgb_frames[i].append(frame)

                if dones[i]:
                    endeds[i] = True
                    ep_id = current_episodes[i].episode_id
                    stats_episodes[ep_id] = infos[i]

                    if config.use_pbar:
                        pbar.update()
                    else:
                        logger.info(log_str.format(evaluated=len(stats_episodes), total=num_eps, time=round(time.time() - start_time)))

                    if len(config.VIDEO_OPTION) > 0:
                        generate_video(video_option=config.VIDEO_OPTION,
                                       video_dir=config.VIDEO_DIR,
                                       images=rgb_frames[i],
                                       episode_id=ep_id,
                                       checkpoint_idx=checkpoint_index,
                                       metrics={"spl": stats_episodes[ep_id]["spl"]},
                                       tb_writer=writer)
                        
                        del stats_episodes[ep_id]["top_down_map_vlnce"]

                    if "GLOBAL_GPS_SENSOR" in config.TASK_CONFIG.TASK.SENSORS:
                        save_agent_trajectory(agent_positions[i], agent_yaws[i], current_episodes[i].scene_id, current_episodes[i].episode_id, config.TRAJECTORY_LOGS_DIR)

            if endeds.all():
                observations = envs.reset()
                current_episodes = envs.current_episodes()

                envs_to_pause = []
                for i in range(envs.num_envs):
                    if (current_episodes[i].episode_id in stats_episodes):
                        envs_to_pause.append(i)

                (envs, observations) = self._pause_envs(envs_to_pause, envs, observations)
                if envs.num_envs == 0: break
                
                observations = extract_position_yaw(observations, config.TASK_CONFIG.TASK.GLOBAL_GPS_SENSOR_UUID)
                processed_observations = process_observations(copy.deepcopy(observations))
                batch = batch_obs(processed_observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                hist_embeds.clear()
                hist_embed = self.policy.module.net(mode="history", observations=batch) # [B, 768]
                while len(hist_embeds) < hist_embeds.maxlen:
                    hist_embeds.append(hist_embed)

                step_id          = -1
                hist_lens        = [hist_embeds.maxlen for _ in range(envs.num_envs)]
                endeds           = np.array([False for _ in range(envs.num_envs)])
                rgb_frames       = [[] for _ in range(envs.num_envs)]
                selected_actions = torch.zeros(envs.num_envs, 1, dtype=torch.long, device=self.device)

                if "GLOBAL_GPS_SENSOR" in config.TASK_CONFIG.TASK.SENSORS:
                    agent_positions = [[] for _ in range(envs.num_envs)]
                    agent_yaws      = [[] for _ in range(envs.num_envs)]

        envs.close()
        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (sum(v[k] for v in stats_episodes.values()) / num_episodes)

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        checkpoint_num = checkpoint_index + 1
        for k, v in aggregated_stats.items():
            logger.info(f"{k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)

    def inference(self) -> None:
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")

        config = self.config.clone()
        if self.config.INFERENCE.USE_CKPT_CONFIG:
            ckpt = self.load_checkpoint(checkpoint_path, map_location="cpu")
            config = self._setup_eval_config(ckpt["config"])

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        config.TASK_CONFIG.DATASET.LANGUAGES = config.INFERENCE.LANGUAGES
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (-1)
        config.IL.ckpt_to_load = config.INFERENCE.CKPT_PATH
        config.TASK_CONFIG.TASK.MEASUREMENTS = []
        config.TASK_CONFIG.TASK.SENSORS = [s for s in config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        config.ENV_NAME = "VLNCEInferenceEnv"
        config.freeze()

        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        observation_space, action_space = self._get_spaces(config, envs=envs)
        self._initialize_policy(config, load_from_ckpt=True, observation_space=observation_space, action_space=action_space)
        self.policy.eval()

        observations = envs.reset()
        current_episodes = envs.current_episodes()
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        step_id = -1
        endeds = np.array([False for _ in range(envs.num_envs)])
        hist_lens = [1 for _ in range(envs.num_envs)]
        selected_actions = torch.zeros(envs.num_envs, 1, dtype=torch.long, device=self.device)
        hist_embeds = [self.policy.module.net(mode='history').expand(envs.num_envs, -1)]

        episode_predictions = defaultdict(list)
        for i in range(envs.num_envs):
            episode_predictions[current_episodes[i].episode_id].append(envs.call_at(i, "get_info", {"observations": {}}))

        pbar = tqdm.tqdm(total=sum(envs.count_episodes()))
        while envs.num_envs > 0:
            with torch.no_grad():
                model_inputs = {"mode": "observation",
                                "observations": batch,
                                "hist_embeds": hist_embeds,
                                "hist_lens": hist_lens,
                                "deterministic": not config.INFERENCE.SAMPLE}

                visual_embed_spatial, actions = self.policy.module.act(**model_inputs)
                selected_actions.copy_(actions)

                step_id += 1 
                model_inputs = {"mode": "history",
                                "step_id": step_id,
                                "observations": visual_embed_spatial,
                                "selected_action": selected_actions}

                hist_embed = self.policy.module.net(**model_inputs)
                hist_embeds.append(hist_embed) 
                hist_lens = [hist_len + 1 for hist_len in hist_lens]

            outputs = envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            for i in range(envs.num_envs):
                if endeds[i]: continue

                episode_predictions[current_episodes[i].episode_id].append(infos[i])

                if dones[i]:
                    endeds[i] = True
                    pbar.update()

            if endeds.all():
                step_id = -1
                observations = envs.reset()
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                hist_embeds = [self.policy.module.net(mode='history').expand(envs.num_envs, -1)] # this is the t=0 step, so there is no history.

                envs_to_pause = []
                current_episodes = envs.current_episodes()
                for i in range(envs.num_envs):
                    if (current_episodes[i].episode_id in episode_predictions):
                        envs_to_pause.append(i)
                    else:
                        episode_predictions[current_episodes[i].episode_id].append(envs.call_at(i, "get_info", {"observations": {}}))

                (envs, batch, hist_embeds, observations) = self._pause_envs(envs_to_pause, envs, batch, hist_embeds, observations)
                if envs.num_envs == 0: break
                
                endeds = np.array([False for _ in range(envs.num_envs)])
                hist_lens = [1 for _ in range(envs.num_envs)]
                selected_actions = torch.zeros(envs.num_envs, 1, dtype=torch.long, device=self.device)

        envs.close()
        pbar.close()

        with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
            json.dump(episode_predictions, f, indent=4)

        logger.info(f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}")

    def observation_logging(self, observation, step_id):
        rgb_image = observation["rgb"]
        rgb_image = rgb_image[..., 0:3][..., ::-1] # change first to RGB and then change to BGR
        rgb_image = np.ascontiguousarray(rgb_image)

        hint_position = observation["hint_position"]
        top_left_corner = (hint_position[0], hint_position[1])
        bottom_right_corner = (hint_position[2], hint_position[3])
        cv2.rectangle(rgb_image, top_left_corner, bottom_right_corner, (0, 255, 0), 2)
        cv2.imwrite(f"/home/nusssun/NAS/Jasun.d/point_where_go_where/data/image_logs/rgb_image_{step_id}.png", rgb_image)
