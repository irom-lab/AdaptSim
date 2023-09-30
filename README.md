# AdaptSim
AdaptSim: Task-Driven Simulation Adaptation for Sim-to-Real Transfer

Conference on Robot Learning (CoRL), 2023

<!-- <img src="teaser.png" alt="drawing" width="88%"/> -->

[[Webpage]](https://irom-lab.github.io/AdaptSim) | [[Arxiv]](https://arxiv.org/abs/2302.04903)

[Allen Z. Ren](https://allenzren.github.io/)<sup>1</sup>,
[Hongkai Dai](https://hongkai-dai.github.io/)<sup>2</sup>,
[Benjamin Burchfiel](http://www.benburchfiel.com/)<sup>2</sup>,
[Anirudha Majumdar](https://irom-lab.princeton.edu/majumdar/)<sup>1</sup>

<sup>1</sup>Princeton University
<sup>2</sup>Toyota Research Institute

Please raise an issue or reach out at allen dot ren at princenton dot edu if you need help with running the code.

## Installation

Install the conda environment with dependencies (tested with Ubuntu 20.04):
```console
conda env create -f environment_linux.yml
pip install -e .
```

AdaptSim is based on [Drake](https://drake.mit.edu), a physically-realistic simulator suited for contact-rich manipulation tasks. Follow the Source Installation instructions [here](https://drake.mit.edu/from_source.html), but use the branch [here](https://github.com/allenzren/drake/tree/adaptsim-release) instead of the Master one.


## Usage

Test the double pendulum, pushing, and scooping environment.
```bash
python test/test_double_pendulum_linearized.py
python test/test_push.py --gui
python test/test_scoop.py --gui
```

We also provide other environments including `AcrobotEnv`, `PendulumEnv`, and `DoublePendulumEnv` that are not used in the paper.

Open Drake Visualizer / Meldis for visualization (especially the contacts).
```bash
cd ~/drake && bazel run //tools:drake_visualizer
cd ~/drake && bazel run //tools:meldis
```

Train the adaptation policies.
```bash
python script/run_adapt.py -cf cfg/pretrain_dp.yaml
python script/run_adapt_trainer.py -cf cfg/pretrain_push.yaml
python script/run_adapt_trainer.py -cf cfg/pretrain_scoop.yaml
```

## Acknowledgement

[Russ Tedrake's manipulation course materials](https://github.com/RussTedrake/manipulation)

[Ben Agro's Panda environment in Drake](https://github.com/BenAgro314/drake_workspace)

## Citation

```
@inproceedings{ren2023adaptsim,
title = {AdaptSim: Task-Driven Simulation Adaptation for Sim-to-Real Transfer},
author = {Ren, Allen Z. and Dai, Hongkai and Burchfiel, Benjamin and Majumdar, Anirudha},
booktitle={Proceedings of the Conference on Robot Learning (CoRL)},
year = {2023},
}
```