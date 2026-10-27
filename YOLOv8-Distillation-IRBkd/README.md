# IRB-KD: YOLOv8 core source snapshot

This is a partial source release of the IRB-KD implementation and its YOLOv8 training integration in V8-DIS. 
The local Ultralytics version is 8.0.114.

The snapshot contains the IRB losses, configuration, detector hooks and trainer integration. It is an overlay for the modified Ultralytics training framework, not a standalone replacement for the published Ultralytics package.

## Files

| File | Purpose |
| --- | --- |
| `ultralytics/nn/modules/irb_loss.py` | Interaction, Reshuffling and Boundary losses. |
| `ultralytics/nn/modules/irb_distillation.py` | Configuration, class counts, teacher-positive assignment and raw Detect outputs. |
| `ultralytics/nn/modules/distillation_loss.py` | IRB-only feature hooks and loss composition. |
| `ultralytics/yolo/engine/trainer.py` | Frozen teacher, optimizer, AMP/DDP and checkpoint integration. |
| `train-distillation.py` | Training entry point. |
| `ultralytics/nn/modules/irb_kd.yaml` | IRB settings. |
| `hyp.yaml` | Common detector hyperparameters from the current VisDrone run. |

## Implementation details

The teacher and student use matching classes, strides and native 16-bin DFL outputs. Boundary alignment uses the L1 distance between cumulative distributions with temperature 0.4. Interaction uses temperature 0.5. The loss weights are alpha=0.00004, beta=0.8 and gamma=0.00006. Frequency ranking with rho_top=rho_tail=0.2 gives a 2/6/2 class partition on VisDrone.

The selected backbone layers have strides 4/8/16/32. RoIAlign uses a 7x7 output, aligned=False and sampling_ratio=-1. Each projection has 128 output channels, giving a flattened cross-layer node embedding of dimension 128*7*7=6272. Teacher and student share each cross-layer projection; the teacher path is detached. Teacher TaskAlignedAssigner uses topk=13, alpha=0.5 and beta=6.0. Reshuffling uses teacher-assigned positive prediction locations and aggregates group densities across the global training batch. Foreground masks use clipped floor/ceil regions; RoIAlign receives continuous ground-truth box coordinates.

## Current run settings

The current VisDrone experiment uses YOLOv8x as teacher and YOLOv8n as student, image size 640, training 300 epochs, SGD, total batch 64 across four GPUs.
