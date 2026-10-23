# IRB-KD: Exploiting Diverse Distilled Knowledge to Unleash UAV Detector Performance

This repository releases the available training source, model and dataset configurations, and evaluation utilities for the project. It includes a modified Ultralytics implementation; the version marker in `ultralytics/__init__.py` is `8.0.114`.

## Release status

The current snapshot is **not yet a complete, verified reproduction of the paper**. The default distiller names, `irb-feature` and `irb-logit`, in `ultralytics/nn/modules/distillation_loss.py` have no corresponding implemented dispatch branches and raise `NotImplementedError`. The mapping between the paper's methods and the available loss implementations, and the complete experiment-specific distillation coefficients, still require verification. No equivalence between `DMKD` and `IRB-KD` is established by this release.

`configs/archive/runs/` contains 13 archived `args.yaml` files from existing experiment records. These preserve recorded settings, with machine-specific paths made portable. They do not contain all teacher checkpoints, student initialization details, or distillation coefficients needed to reproduce the experiments. `RDR-KD` is an archived experiment name, not a confirmed mapping to `IRB-KD`.

## Files

| Path | Purpose |
| --- | --- |
| `train.py` | Teacher training and ordinary student training |
| `train-distillation.py` | Student training with a teacher model |
| `ultralytics/` | Modified framework, model definitions, trainer, and losses |
| `ultralytics/nn/modules/distillation_loss.py` | Feature/logit distiller selection, feature hooks, and loss aggregation |
| `ultralytics/nn/modules/distillation_build.py`, `kd_loss.py` | Available distillation loss implementations |
| `ultralytics/models/v8/` | YOLOv8 model architecture configurations and scales |
| `ultralytics/datasets/` | Dataset YAML configurations |
| `hyp.yaml` | Base optimization and augmentation hyperparameters |
| `configs/archive/runs/` | Archived experiment arguments, with the limitations above |
| `val.py`, `coco_evaluate.py` | Detector validation and COCO-format evaluation |
| `visdrone_coco_format.py`, `YOLO_predictions2coco.py` | Annotation conversion and prediction image-ID alignment |

## Installation

Install a compatible PyTorch/torchvision pair for your system, then install this repository in a separate environment:

```bash
git clone https://github.com/Golden-Ele/IRB-KD.git
cd IRB-KD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Use the local modified `ultralytics/` package for training. Replacing it with a stock PyPI Ultralytics package omits this project's distillation changes. The dependency list is a starting point; a complete training run has not been used to validate a Python/PyTorch/CUDA environment for this release.

## Data preparation

Obtain public datasets from their respective providers and prepare YOLO-format images and labels. The included YAML files describe the expected split directories and class names. Set each dataset YAML's `path` to the **absolute path** of your local dataset root. Relative paths in this framework are resolved against the Ultralytics `datasets_dir` setting, which may differ from your repository directory.

- `ultralytics/datasets/VisDrone.yaml`: VisDrone train, validation, and test-dev splits.
- `ultralytics/datasets/UAVDT.yaml`: the recorded configuration uses `UAVDT-test/images` for both `val` and `test`. These are the same images, not independent splits.
- `ultralytics/datasets/AU-AIR.yaml`: expects `train/images`, `val/images`, and `test/images`. The original filename-level split manifests are not included, so this YAML alone cannot reconstruct the exact experimental split.
- `ultralytics/datasets/coco.yaml`: COCO image and annotation paths.

The self-captured dataset cannot be publicly released or provided on request. Images and annotations from that dataset are excluded from this repository; its results cannot be independently reproduced from the public release alone.

## Training

The following commands illustrate the interfaces; their batch sizes and run names are examples, not recovered paper experiment settings. Adjust paths, GPU selection, and batch size to your environment. The loader resolves scale-specific names such as `yolov8n.yaml` and `yolov8x.yaml` through the shared `yolov8.yaml` definition.

Teacher training:

```bash
python train.py \
  --yaml ultralytics/models/v8/yolov8x.yaml \
  --data ultralytics/datasets/VisDrone.yaml --cfg hyp.yaml \
  --epochs 300 --batch 16 --imgsz 640 --device 0 \
  --project runs/teacher --name visdrone-yolov8x
```

Ordinary student training:

```bash
python train.py \
  --yaml ultralytics/models/v8/yolov8n.yaml \
  --data ultralytics/datasets/VisDrone.yaml --cfg hyp.yaml \
  --epochs 300 --batch 16 --imgsz 640 --device 0 \
  --project runs/student --name visdrone-yolov8n
```

`--weight /path/to/checkpoint.pt` optionally supplies initialization weights. Trained checkpoints are written below the selected `--project` and `--name`; checkpoints are not bundled here.

The `train-distillation.py` entry point requires `--teacher-weight /path/to/teacher/best.pt`; `--weight` selects the student's initialization checkpoint. Run `python train-distillation.py --help` to inspect the remaining arguments. **Distillation training will not complete with the unresolved default distiller names described above.** Verify the intended paper method and its settings before using this entry point for an experiment.

Teacher/student feature channels and tapped layers are currently hardcoded in `distillation_loss.py`; changing architectures requires checking their compatibility. `hyp.yaml` and archived `args.yaml` files do not provide a complete paper-specific distillation configuration.

## Evaluation

Validate a checkpoint and save JSON predictions:

```bash
python val.py \
  --weight /path/to/best.pt --data ultralytics/datasets/VisDrone.yaml \
  --split val --imgsz 640 --batch 16 \
  --project runs/val --name visdrone --save_json
```

Evaluate predictions against COCO-format ground truth, using the actual prediction JSON path produced by validation:

```bash
python coco_evaluate.py \
  --annotations /path/to/annotations.json \
  --predictions /path/to/predictions.json
```

When prediction image IDs need alignment with an existing COCO annotation file:

```bash
python YOLO_predictions2coco.py \
  --predictions /path/to/predictions.json \
  --annotations /path/to/annotations.json \
  --output /path/to/aligned-predictions.json
```

Convert original VisDrone TXT annotations to COCO format:

```bash
python visdrone_coco_format.py \
  --dataset visdrone \
  --images /path/to/VisDrone2019-DET-val/images \
  --annotations /path/to/VisDrone2019-DET-val/annotations \
  --output /path/to/converted --split val
```

The output is `Anno/VisDrone2019-DET_val_coco.json` below `--output`. `--dataset uavdt` selects the UAVDT converter and writes `Anno/UAVDT_<split>_coco.json`.

The VisDrone converter removes categories 0 and 11 and annotations with truncation or occlusion values greater than 1. It **retains boxes smaller than 4 pixels**: the small-box branch does not skip them. The UAVDT converter instead removes annotations with `score == 0`, truncation greater than 1, or occlusion greater than 1, and shifts category IDs by 1. Converted image IDs are filename stems. These filtering rules affect evaluation and should be kept consistent when comparing results.

## License

The bundled Ultralytics source retains its AGPL-3.0 notices. See `LICENSE` for the license included with this release.
