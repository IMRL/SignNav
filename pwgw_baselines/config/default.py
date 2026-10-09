from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import habitat_baselines.config.default
from habitat.config.default import Config as CN
from habitat.config.default import CONFIG_FILE_SEPARATOR

from habitat_extensions.config.default import get_extended_config as get_task_config


# ----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# ----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/pwgw_baselines/pwgw_task.yaml"
_C.TASK_CONFIG           = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS     = []  # store command line options as list of strings
_C.TRAINER_NAME          = "dagger_trainer"
_C.ENV_NAME              = "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_IDS     = [0]
_C.VIDEO_OPTION          = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR             = "data/video_logs"
_C.CHECKPOINT_FOLDER     = "data/checkpoints/cma"
_C.RESULTS_DIR           = "data/checkpoints/cma/evals"
_C.TENSORBOARD_DIR       = "data/tensorboard_dirs/cma"
_C.IMAGE_LOGS_DIR        = "data/image_logs"
_C.TRAJECTORY_LOGS_DIR   = "data/trajectory_logs"


# ----------------------------------------------------------------------------
# EVAL CONFIG
# ----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.SPLIT            = "val_seen"
_C.EVAL.EPISODE_COUNT    = -1
_C.EVAL.SAMPLE           = False
_C.EVAL.USE_CKPT_CONFIG  = True
_C.EVAL.SAVE_RESULTS     = True
_C.EVAL.EVAL_NONLEARNING = False

_C.EVAL.NONLEARNING = CN()
_C.EVAL.NONLEARNING.AGENT = "RandomAgent"


# ----------------------------------------------------------------------------
# INFERENCE CONFIG
# ----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT                 = "test"
_C.INFERENCE.SAMPLE                = False
_C.INFERENCE.FORMAT                = "rxr"  # either 'rxr' or 'r2r'
_C.INFERENCE.USE_CKPT_CONFIG       = True
_C.INFERENCE.CKPT_PATH             = "data/checkpoints/CMA_PM_DA_Aug.pth"
_C.INFERENCE.PREDICTIONS_FILE      = "predictions.json"
_C.INFERENCE.INFERENCE_NONLEARNING = False

_C.INFERENCE.NONLEARNING = CN()
_C.INFERENCE.NONLEARNING.AGENT = "RandomAgent"


# ----------------------------------------------------------------------------
# IMITATION LEARNING CONFIG
# ----------------------------------------------------------------------------
_C.IL = CN()
_C.IL.lr = 2.5e-4
_C.IL.batch_size = 5
# number of network update rounds per iteration
_C.IL.epochs = 4
# if true, uses class-based inflection weighting
_C.IL.use_iw = True
# inflection coefficient for RxR training set GT trajectories (guide): 1.9
# inflection coefficient for R2R training set GT trajectories: 3.2
_C.IL.inflection_weight_coef = 2.0
# load an already trained model for fine tuning
_C.IL.load_from_ckpt = False
_C.IL.ckpt_to_load = "data/checkpoints/cma/ckpt.0.pth"
# if True, loads the optimizer state, epoch, and step_id from the ckpt dict.
_C.IL.is_requeue = False


# ----------------------------------------------------------------------------
# IL: RECOLLECT TRAINER CONFIG
# ----------------------------------------------------------------------------
_C.IL.RECOLLECT_TRAINER = CN()
_C.IL.RECOLLECT_TRAINER.preload_trajectories_file = False
_C.IL.RECOLLECT_TRAINER.trajectories_file = ("data/trajectories_dirs/debug/trajectories.json.gz")
# if set to a positive int, episodes with longer paths are ignored in training
_C.IL.RECOLLECT_TRAINER.max_traj_len = -1
# if set to a positive int, effective_batch_size must be some multiple of IL.batch_size.
# Gradient accumulation enables an arbitrarily high "effective" batch size.
_C.IL.RECOLLECT_TRAINER.effective_batch_size = -1
# _C.IL.RECOLLECT_TRAINER.preload_size = 30 # by author
_C.IL.RECOLLECT_TRAINER.preload_size = 10 # by sunjian


# ----------------------------------------------------------------------------
# IL: DAGGER CONFIG
# ----------------------------------------------------------------------------
_C.IL.DAGGER = CN()
# dataset aggregation rounds (1 for teacher forcing)
_C.IL.DAGGER.iterations = 10
_C.IL.DAGGER.start_iteration = 0
# episodes collected per iteration (size of dataset for teacher forcing)
_C.IL.DAGGER.update_size = 5000
# probability of taking the expert action (1.0 for teacher forcing)
_C.IL.DAGGER.p = 0.75
_C.IL.DAGGER.expert_policy_sensor_uuid = "reference_path_sensor"
# _C.IL.DAGGER.expert_policy_sensor_uuid = "shortest_path_sensor"
_C.IL.DAGGER.lmdb_map_size = 2.4e12 
# if True, saves data to disk in fp16 and converts back to fp32 when loading.
_C.IL.DAGGER.lmdb_fp16 = False
# How often to commit the writes to the DB, less commits is better, but everything must be in memory until a commit happens.
_C.IL.DAGGER.lmdb_commit_frequency = 500 # original value: 500
# If True, load precomputed features directly from lmdb_features_dir.
_C.IL.DAGGER.preload_lmdb_features = False
_C.IL.DAGGER.lmdb_features_dir = ("data/trajectories_dirs/debug/trajectories.lmdb")
_C.IL.DAGGER.drop_existing_lmdb_features = True


# ----------------------------------------------------------------------------
# MODEL CONFIG
# ----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.policy_name = "PWGWPolicy"
_C.MODEL.max_hist_len = 10
_C.MODEL.hidden_size = 768
_C.MODEL.vit_input_size = 224
_C.MODEL.vit_patch_size = 32
_C.MODEL.vit_embed_dim = 768
_C.MODEL.max_action_steps = 100
_C.MODEL.hidden_dropout_prob = 0.1
_C.MODEL.attention_probs_dropout_prob = 0.1
_C.MODEL.pred_head_dropout_prob = 0.1
_C.MODEL.num_t_layers = 6 # 6
_C.MODEL.num_attention_heads = 6 # 12
_C.MODEL.output_attentions = True
_C.MODEL.layer_norm_eps = 1e-12
_C.MODEL.intermediate_size = 3072
_C.MODEL.hidden_act = "gelu"
_C.MODEL.ignore_id = -100
_C.MODEL.attention_weight = 0.5
_C.MODEL.episode_length = 0 # 0 means using the entire episode length!
# RGB ENCODER
_C.MODEL.RGB_ENCODER = CN()
_C.MODEL.RGB_ENCODER.stem_channel = 3
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.output_size = 256
_C.MODEL.RGB_ENCODER.trainable = True
# DEPTH ENCODER
_C.MODEL.DEPTH_ENCODER = CN()
_C.MODEL.DEPTH_ENCODER.stem_channel = 1
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 128
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50" # type of resnet to use
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = ("data/ddppo-models/gibson-2plus-resnet50.pth") # path to DDPPO resnet weights
_C.MODEL.DEPTH_ENCODER.trainable = True
# HINT ENCODER
_C.MODEL.HINT_ENCODER = CN()
_C.MODEL.HINT_ENCODER.input_type = "image" # "image" / "language"
_C.MODEL.HINT_ENCODER.language_vocab_size = 4
_C.MODEL.HINT_ENCODER.output_size = 256
_C.MODEL.HINT_ENCODER.resnet_output_size = 256
_C.MODEL.HINT_ENCODER.trainable = True


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


def get_config(config_paths: Optional[Union[List[str], str]] = None, opts: Optional[list] = None) -> CN:
    """Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    
    return config
