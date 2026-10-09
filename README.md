# SignNav: Directional Signage-Cue Guided Visual Navigation in Large-Scale Indoor Environments

Humans routinely leverage directional cues from signage to navigate toward destinations in unfamiliar LargeScale Indoor (LSI) environments, such as hospitals and airport
terminals. However, learning how embodied agents can use sparse and target-relevant signage cues for closed-loop navigation remains underexplored. This work studies SignNav, a directional signage-cue guided visual navigation task in which an agent
navigates to a semantic target using egocentric observations together with directional cues and cue locations provided by an upstream sign parser. This formulation focuses on downstream cue-conditioned policy learning, rather than open-ended signage parsing, OCR, or compound sign reasoning. To facilitate research in this domain, we construct the LSI-Dataset, a largescale benchmark with signage cues-labeled indoor environments
and automatically generated trajectory-action episodes. SignNav presents two key challenges: grounding transient directional cues into the current observation, and maintaining temporal memory when signage is sparse or temporarily unavailable. To address these challenges, we propose the Spatial-Temporal Aware Transformer (START), which combines a spatial-aware module for cueobservation grounding with a temporal-aware module for historybased decision making.

<p align="center">
  <img width="775" height="360" src="./data/demo.mp4" alt="Real-world Demo">
</p>

## Setup

This project is developed with Python 3.6. If you are using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/), you can create an environment:

```bash
conda create -n signnav python=3.6
conda activate signnav
```

SignNav uses [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) 0.1.7 which can be [built from source](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7#installation) or installed from conda:

```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
```

Then install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7):

```bash
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
# installs both habitat and habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

Now you can install SignNav:

```bash
git clone git@github.com:IMRL/SignNav.git
cd SignNav
python -m pip install -r requirements.txt
```

### Dataset

We will release the LSI-Dataset as soon as possible.

## Starter Code

The `run.py` script controls training and evaluation for all models and datasets:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 run.py --exp-config pwgw_baselines/config/pwgw_task/pwgw.yaml --run-type {train | eval | inference}
```

For lists of modifiable configuration options, see the default [task config](habitat_extensions/config/default.py) and [experiment config](pwgw_baselines/config/default.py) files.

### Training Agents

The `DaggerTrainer` class is the standard trainer and supports teacher forcing or dataset aggregation (DAgger). This trainer saves trajectories consisting of RGB, depth, ground-truth actions, and instructions to disk to avoid time spent in simulation.

### Evaluating Agents

Evaluation on validation splits can be done by running `python run.py --exp-config path/to/experiment_config.yaml --run-type eval`. If `EVAL.EPISODE_COUNT == -1`, all episodes will be evaluated. If `EVAL_CKPT_PATH_DIR` is a directory, each checkpoint will be evaluated one at a time.

### Cuda

Cuda will be used by default if it is available. We find that one GPU for the model and several GPUs for simulation is favorable.

```yaml
SIMULATOR_GPU_IDS: [0]  # list of GPU IDs to run simulations
TORCH_GPU_ID: 0  # GPU for pytorch-related code (the model)
NUM_ENVIRONMENTS: 1  # Each GPU runs NUM_ENVIRONMENTS environments
```

The simulator and torch code do not need to run on the same device. For faster training and evaluation, we recommend running with as many `NUM_ENVIRONMENTS` as will fit on your GPU while assuming 1 CPU core per env.

## License

The SignNav codebase is [MIT licensed](LICENSE). Trained models and task dataset are distributed under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).

## Citing

To cite our SignNav, please cite the following paper [SignNav](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11676065):

```tex
@ARTICLE{11676065,
  author={Sun, Jian and Huang, Yuming and Li, He and Xiao, Shuqi and Guo, Shenyan and Ghaffari, Maani and Li, Qingbiao and Xu, Chengzhong and Kong, Hui},
  journal={IEEE Robotics and Automation Letters}, 
  title={SignNav: Directional Signage-Cue Guided Visual Navigation in Large-Scale Indoor Environments}, 
  year={2026},
  volume={11},
  number={11},
  pages={12304-12311},
  doi={10.1109/LRA.2026.3730302}}
```
