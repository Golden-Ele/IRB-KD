# IRB-KD

IRB-KD combines Interaction, Reshuffling and Boundary distillation for object detection.

| Directory | Contents | Framework |
| --- | --- | --- |
| [YOLOv8-Distillation-IRBkd](YOLOv8-Distillation-IRBkd/) | Selected losses, detector hooks, trainer integration and configurations. | Modified Ultralytics 8.0.114 |
| [FasterRCNN-Distillation-IRBkd](FasterRCNN-Distillation-IRBkd/) | IRB losses, learned distributional RoI head, training entry point and VisDrone configuration. | MMDetection 2.28.2 / MMCV-full 1.7.2 |

See the [YOLOv8 README](YOLOv8-Distillation-IRBkd/README.md) and [Faster R-CNN README](FasterRCNN-Distillation-IRBkd/README.md) for file descriptions, dependencies and configuration details.

The implementations share the three distillation objectives and adapt their prediction sampling and boundary representation to each detector. YOLOv8 uses its native 16-bin head; Faster R-CNN uses a learned 17-point boundary head on sampled RoIs.

This is a selected source release. 
The YOLOv8 directory is an overlay for the modified training framework; 
the Faster R-CNN directory is a compact MMDetection extension.
