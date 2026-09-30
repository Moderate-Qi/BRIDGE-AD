# BRIDGE-AD

Official repository for **BRIDGE-AD: Bipolar Relation-Instructed Diffusion Generation of Anomalous and Normal Examples for Anomaly Detection**.

## Overview

BRIDGE-AD generates both anomalous images and normal images with varied appearances for industrial anomaly detection. Instructions based on normal component relations specify where to edit and whether to violate or preserve those relations. Separate diffusion control branches learn the two editing objectives, while structural correction helps maintain the intended local structure and preserve surrounding content.

## Release Status

This is a **partial release of the training workflow**. It includes the main training launcher and a training-update function, showing how the training stages and module interfaces are organized.

**The majority of the remaining code, including the core module implementations and their internal training dependencies, will be released upon acceptance of the paper.**

## Datasets

Download the three benchmarks from their original providers and cite their corresponding publications:

| Dataset | Original source | Reference |
| --- | --- | --- |
| MVTec AD | [Official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | Bergmann et al., CVPR 2019 |
| VAD (Valeo Anomaly Dataset) | [Authors' repository](https://github.com/abc-125/segad) | Baitieva et al., CVPR 2024 |
| MVTec LOCO AD | [Official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) | Bergmann et al., IJCV 2022 |

See [dataset references](DATASETS.md) for the full citations. Dataset images are not redistributed in this repository. Their original licenses apply independently of this repository's code license.

## Training Workflow

The launcher follows the training order used by the implementation:

1. Prepare the normal-appearance control branch for the selected dataset.
2. Train the HP and HN branches with the base objective.
3. Continue with bipolar training, using fixed parent checkpoints for the paired HP/HN updates within each round.

HP denotes anomalous examples; HN denotes normal examples with varied appearances. The normal-appearance branch and diffusion backbone are frozen during the subsequent polarity-branch training.

`train.py` launches the internal training stages in this order and stops if a stage fails. `training_step.py` exposes a base-objective update: sample a diffusion timestep and noise, obtain the conditioned noise prediction, compute the loss, and update the active control branch. The model construction, condition encoding, loss definitions, bipolar objectives, and generation logic remain in the unreleased internal modules.

### Inspect the Public Entry Point

Python 3.10 or newer is sufficient for these two inspection commands:

```bash
python train.py --help
python train.py --show-workflow
```

Neither command imports a model, downloads assets, or starts training.

### Launch Training With the Internal Dependencies

The following commands require the internal `bridge_ad` package, its environment, and a complete training plan. **Those dependencies and the plan are scheduled for the post-acceptance release.**

1. Obtain the datasets from the links above and prepare the training/support manifests.
2. Install the training environment and the `bridge_ad` implementation.
3. Set the dataset, pretrained backbone, manifest, checkpoint, and output paths in the training plan.
4. Inspect the resolved stage commands, then launch training:

```bash
python train.py --config /path/to/training_plan.json --dry-run
python train.py --config /path/to/training_plan.json
```

The plan supplies argument lists for `texture`, `base_hp`, `base_hn`, and the paired entries in `bipolar_rounds`. Each bipolar round refers to fixed parent checkpoints; completing its HP job must not change the parent used by its HN job. Hyperparameters and asset preparation are provided by the internal configuration, not inferred by this launcher.

## Repository Contents

```text
train.py             Main training workflow and stage launcher
training_step.py     Base-objective training update and batch interface
DATASETS.md          Original dataset sources and citations
LICENSE              MIT license for the released code
```

## License

The code currently included in this repository is released under the [MIT License](LICENSE). External datasets, pretrained models, and third-party components remain subject to their respective licenses; they are not redistributed here.
