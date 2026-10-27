# IRB-KD: Faster R-CNN core source snapshot

A compact MMDetection 2.28.2 extension for IRB-KD. 
The configuration uses a ResNet-101/FPN teacher and a ResNet-50/FPN student. 
Both use a learned distributional RoI localization head.

## Files

| File | Purpose |
| --- | --- |
| `irb_kd/irb_loss.py` | Foreground interaction, instance relations and frequency-group reshuffling. |
| `irb_kd/boundary.py` | Proposal-relative distances, support points, DFL supervision and W1/KL boundary losses. |
| `irb_kd/distribution_bbox_head.py` | Learned 17-point localization head, regression supervision and inference decoding. |
| `irb_kd/roi_head.py` | RoIs, assigned labels and student predictions during training. |
| `irb_kd/faster_rcnn.py` | Frozen teacher, shared RoI evaluation, checkpoint validation and detector integration. |
| `irb_kd/distillation.py` | IRB configuration, class counts, feature hooks and loss composition. |
| `irb_kd/optim.py` | Trainable-parameter filtering, finite-value checks and iteration resume state. |
| `irb_kd/__init__.py` | Registers the detector, heads and hooks with MMDetection. |
| `train.py` | Training entry point, configuration overrides and checkpoint/protocol checks. |
| `configs/irb_visdrone.py` | Self-contained model, dataset, optimizer and distillation configuration. |

`requirements.txt` lists the dependencies.

## Environment

The source was checked with Python 3.9, PyTorch 2.0.0, CUDA 11.8, MMCV-full 1.7.2 and MMDetection 2.28.2. A matching installation can be prepared with:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
```

## Data

The configuration expects VisDrone2019 images and COCO-format annotations:

```text
data/VisDrone/
  annotations/train.json
  annotations/val.json
  annotations/test-dev.json
  VisDrone2019-DET-train/images/
  VisDrone2019-DET-val/images/
  VisDrone2019-DET-test-dev/images/
```

The class order is `pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`. The annotation category order must match this tuple. The local conversion excluded categories 0 and 11, annotations with score 0 and boxes with nonpositive dimensions. Use the same annotation policy for the baseline, teacher and distilled student.

## Training

The common configuration uses 24 epochs, two images per GPU, SGD with learning rate 0.01, momentum 0.937 and weight decay 0.0005, and seed 0. 
Learning-rate steps are at epochs 16 and 22.

Train the R50 student without distillation:

```bash
torchrun --standalone --nproc_per_node=4 train.py configs/irb_visdrone.py \
  --launcher pytorch --work-dir work_dirs/r50_baseline \
  --cfg-options model.distillation=None model.teacher_model=None model.teacher_checkpoint=None
```

Train the R101 teacher with the same distributional head:

```bash
torchrun --standalone --nproc_per_node=4 train.py configs/irb_visdrone.py \
  --launcher pytorch --work-dir work_dirs/r101_teacher \
  --cfg-options model.distillation=None model.teacher_model=None model.teacher_checkpoint=None \
    model.backbone.depth=101 model.backbone.init_cfg.checkpoint=torchvision://resnet101
```

After teacher training, set `TEACHER_CHECKPOINT` to its selected checkpoint. Joint distillation with the reference loss weights:

```bash
torchrun --standalone --nproc_per_node=4 train.py configs/irb_visdrone.py \
  --launcher pytorch --work-dir work_dirs/irb_joint \
  --cfg-options model.teacher_checkpoint="$TEACHER_CHECKPOINT"
```

## Distillation details and controls

The objective is `L_det + alpha * L_I + beta * L_R + gamma * L_B`. Detection supervision includes RPN losses, RoI classification, SmoothL1 regression and DFL. Interaction uses backbone stages C2–C5, relation temperature 0.5, RoIAlign size 7 and embedding channels 128. Reshuffling uses foreground-class logits of positive sampled RoIs and class-frequency groups with `rho_top=rho_tail=0.2`. Its reduction is a sum over positive samples, without a positive-count mean or temperature-squared multiplier.

The teacher evaluates exactly the student's sampled RoIs. The localization head predicts 17 support points `i * d_max / 16`, for `i=0,...,16`, for each box edge and foreground class. The ten-class configuration therefore has 680 regression outputs. Distances are normalized by proposal width/height around its center. DFL supervises adjacent support points; expected distances decode the boxes used for SmoothL1 supervision and inference. Targets outside the support are excluded from DFL and Boundary distillation while retaining classification and SmoothL1 supervision.

Boundary distillation uses temperature 0.4 and normalizes the sum over four edges by the global valid-positive count. W1 sums absolute cumulative-probability differences multiplied by the support spacing. KL computes `KL(p_teacher || p_student)` without a temperature-squared multiplier. DDP averaging is compensated once when composing the global distillation objective.

Append these overrides to the joint training command:

| Objective | `--cfg-options` overrides |
| --- | --- |
| Interaction only | `model.distillation.logic_distiller=None` |
| Reshuffling + Boundary | `model.distillation.feature_distiller=None` |
| Boundary only | `model.distillation.feature_distiller=None model.distillation.beta=0` |
| KL boundary control | `model.distillation.boundary_criterion=kl` |
