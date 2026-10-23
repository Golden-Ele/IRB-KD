# Configuration records

The paper method is **IRB-KD**. These files expose the training arguments that
survived in the project archive. They are incomplete historical records, not
validated commands for reproducing every paper result. In particular, `RDR-KD`
is an original directory label; its equivalence to IRB-KD has not been established
by the configuration files alone. Directory labels for other methods and model
sizes likewise do not prove the runtime method or architecture.

## Included archives

The paths below preserve the original `runs/` layout beneath `configs/archive/`.
All 13 available `args.yaml` files are included. Training values are unchanged;
the absolute dataset-config path in the COCO `8n-coco` record is replaced with its
repository-relative path. Comments identify each original record.

| Dataset | Original record, relative to `runs/` |
| --- | --- |
| VisDrone | `train-distillation-visdrone/RDR-KD/args.yaml` |
| VisDrone | `train-distillation-visdrone/pkd-960/args.yaml` |
| VisDrone | `train-distillation-visdrone/mgd-dkd/args.yaml` |
| VisDrone | `train-teacher/8x-visdrone/args.yaml` |
| UAVDT | `train-distillation-UAVDT/8n/args.yaml` |
| UAVDT | `train-distillation-UAVDT/mgd/args.yaml` |
| UAVDT | `train-teacher/8x-UAVDT/args.yaml` |
| AU-AIR | `train-distillation-AU-AIR/8n/args.yaml` |
| AU-AIR | `train-distillation-AU-AIR/8n2/args.yaml` |
| COCO | `train-distillation-coco/8n-coco/args.yaml` |
| COCO | `train-distillation-coco/mgd/args.yaml` |
| COCO | `train-distillation-coco/awd/args.yaml` |
| COCO | `train-teacher/8x-coco/args.yaml` |

`8n2` retains its original internal `name: 8n`; it is not evidence of a different
random seed. Output paths and checkpoint paths refer to the historical run layout
and do not imply that those outputs are distributed with this repository.

## Information these records do not establish

- Most records contain `model: null`; two contain a `last.pt` checkpoint path.
  Neither form is a complete specification of the initial architecture and
  weights. `pretrained: false` does not establish that a separate script did not
  load an initialization checkpoint before training.
- The arguments do not record the teacher checkpoint, explicit student
  initialization, runtime distiller selection, distillation coefficients, or
  the full correspondence between paper tables and archived runs. These details
  must be supplied from the authors' actual experimental records. Do not fill
  them from current source defaults or method names in directory labels.
- There is no archived AU-AIR teacher `args.yaml`, nor an explicitly named
  IRB-KD/RDR-KD main-method record for UAVDT or AU-AIR among these 13 files.
  This describes the archive coverage and does not assert that those experiments
  were never performed.
- `hyp.yaml` contains the shared detection/optimization/augmentation settings;
  it does not contain a complete distillation configuration. Current entrypoint
  defaults are not a substitute for an archived run's values.

## Dataset paths and splits

Dataset YAML files are under `ultralytics/datasets/`. Machine-specific root paths
have been replaced with `VisDrone`, `UAVDT`, `AU-AIR`, `COCO`, or `DET`. This code
resolves those relative paths against the configured Ultralytics `datasets_dir`;
set that directory for your machine or edit each YAML `path` to your local root.
The existing `train`, `val`, `test`, and class mappings have been preserved.

The archived UAVDT configuration uses `UAVDT-test/images` for both validation and
testing; the training image-count comment is 23,829. This publication does not
replace that historical protocol with a new split. `UAVDT-2.yaml` points to
`UAVDT-val-3` and is a separate alternative configuration, not evidence of the
standard archived experiments using it. `VisDrone-DET.yaml` retains empty training
and test fields and is not a complete training configuration.

The AU-AIR YAML describes the project's train/validation/test directories. The
YAML alone does not reconstruct membership of that study-specific split; exact
split lists or a deterministic preparation procedure still need to be supplied
from the original records before claiming complete split reproducibility.

## Self-captured data template

No verified historical self-captured-data training configuration was identified
in these archives. `self_captured.example.yaml` is deliberately incomplete and
contains no measured settings, real filenames, images, or annotations. The authors
must fill its actual root, split definitions, and ordered class mapping, and
separately document the actual model/training/distillation settings before using
or describing it as a configuration for a reported experiment. Publishing this
template does not make the unavailable self-captured dataset reproducible.
