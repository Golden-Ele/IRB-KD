"""Train a detector with MMDetection's runners, hooks and data loaders."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from mmcv import Config, ConfigDict, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, compat_cfg, find_latest_checkpoint,
                         get_root_logger, replace_cfg_vals,
                         setup_multi_processes, update_data_root)

import irb_kd  # Register the detector, RoI head and optimizer constructor.
from irb_kd.distribution_bbox_head import DistributionShared2FCBBoxHead
from irb_kd.faster_rcnn import validate_head_checkpoint, without_pretraining
from irb_kd.boundary import make_support
from irb_kd.distillation import count_training_classes, load_irb_config


def validate_class_order(dataset):
    """MMDetection labels follow cat_ids, which must agree with the class-name tuple."""
    actual = tuple(dataset.coco.cats[category]['name'] for category in dataset.cat_ids)
    if actual != tuple(dataset.CLASSES):
        raise ValueError('COCO category order differs from dataset.CLASSES; use the same order in both.')
    return actual


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Train a Faster R-CNN detector')
    parser.add_argument('config', help='training config file')
    parser.add_argument('--work-dir')
    parser.add_argument('--resume-from')
    parser.add_argument('--auto-resume', action='store_true')
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--diff-seed', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--auto-scale-lr', action='store_true')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args(argv)


def prepare_distillation(cfg, dataset):
    """Resolve the teacher and class statistics before building the detector."""
    classes = validate_class_order(dataset)
    head = cfg.model.roi_head.bbox_head
    if head.num_classes != len(classes):
        raise ValueError(
            'The detector class count differs from the training dataset.')
    cfg.model.classes = classes
    if head.type == 'DistributionShared2FCBBoxHead':
        make_support(head.num_bins, head.d_max)
        if head.dfl_loss_weight is None:
            raise ValueError(
                'Set model.roi_head.bbox_head.dfl_loss_weight in the experiment config.'
            )
    kd = cfg.model.get('distillation')
    if kd is None:
        return classes
    if head.type != 'DistributionShared2FCBBoxHead':
        raise ValueError(
            'Distillation requires the adapted distributional head.')
    checkpoint = cfg.model.get('teacher_checkpoint')
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError(
            'Set model.teacher_checkpoint to a trained distributional R101 checkpoint.'
        )
    cfg.model.teacher_checkpoint = str(Path(checkpoint).resolve())
    if cfg.model.get('teacher_model') is None:
        teacher_config = cfg.get('teacher_config')
        if not teacher_config:
            raise ValueError('Set teacher_config or model.teacher_model.')
        cfg.model.teacher_model = deepcopy(
            Config.fromfile(teacher_config).model)
    teacher = cfg.model.teacher_model
    if teacher.get('distillation') is not None:
        raise ValueError('teacher_config must describe a standalone teacher.')
    teacher.roi_head.bbox_head.update(num_classes=len(classes),
                                      num_bins=head.num_bins,
                                      d_max=head.d_max,
                                      dfl_loss_weight=head.dfl_loss_weight)
    kd = load_irb_config(kd)
    if kd['logic_distiller'] == 'irb-logit' and kd['beta'] > 0:
        counts = count_training_classes(dataset)
        if kd['class_counts'] is not None and list(
                kd['class_counts']) != counts:
            raise ValueError(
                'Configured class counts differ from the filtered training annotations.'
            )
        kd['class_counts'] = counts
    cfg.model.distillation = kd
    return classes


def training_protocol(cfg, dataset, classes, world_size):
    loader = cfg.data.train_dataloader
    return dict(dataset_sha256=file_sha256(dataset.ann_file),
                classes=classes,
                head=dict(cfg.model.roi_head.bbox_head),
                distillation=deepcopy(cfg.model.get('distillation')),
                teacher_sha256=(file_sha256(cfg.model.teacher_checkpoint)
                                if cfg.model.get('distillation') else None),
                training=dict(seed=cfg.seed,
                              diff_seed=cfg.get('diff_seed', False),
                              total_batch=loader.samples_per_gpu * world_size,
                              optimizer=dict(cfg.optimizer),
                              optimizer_config=dict(cfg.optimizer_config),
                              lr_config=dict(cfg.lr_config),
                              auto_scale_lr=cfg.get('auto_scale_lr'),
                              fp16=cfg.get('fp16'),
                              pipeline=cfg.data.train.pipeline,
                              train_cfg=cfg.model.train_cfg))


def validate_checkpoint(model, cfg, classes, protocol):
    path = cfg.get('resume_from') or cfg.get('load_from')
    if not path:
        return {}
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(model.roi_head.bbox_head, DistributionShared2FCBBoxHead):
        validate_head_checkpoint(model, checkpoint, classes)
    state = {
        key.removeprefix('module.'): value
        for key, value in checkpoint['state_dict'].items()
    }
    if cfg.get('resume_from'):
        saved = checkpoint.get('meta', {}).get('irb_protocol')
        if saved != protocol:
            raise ValueError(
                'Resume configuration, training annotations or teacher differ from the saved protocol.'
            )
        model.load_state_dict(state, strict=True)
        return checkpoint.get('meta', {})
    else:
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(('teacher.', 'distillation.'))
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith(('teacher.', 'distillation.'))
                             for key in missing):
            raise ValueError(
                'Student initialization checkpoint is incompatible with this detector.'
            )
        cfg.load_from = None
        return {}


def train(cfg, args):
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    cfg.work_dir = str(
        Path(args.work_dir or cfg.get('work_dir')
             or Path('work_dirs') / Path(args.config).stem).resolve())
    cfg.device = args.device
    cfg.resume_from = args.resume_from or cfg.get('resume_from')
    cfg.load_from = cfg.get('load_from')
    cfg.auto_resume = args.auto_resume or cfg.get('auto_resume', False)
    if cfg.auto_resume and not cfg.resume_from:
        cfg.resume_from = find_latest_checkpoint(cfg.work_dir)
    if cfg.resume_from and cfg.load_from:
        raise ValueError('Choose either resume_from or load_from.')
    if args.auto_scale_lr:
        cfg.auto_scale_lr = dict(
            enable=True, base_batch_size=cfg.auto_scale_lr.base_batch_size)
    if cfg.get('fp16') is not None and cfg.device != 'cuda':
        raise ValueError('FP16 training requires CUDA.')
    setup_multi_processes(cfg)
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', False)
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    distributed = args.launcher != 'none'
    if distributed:
        if cfg.device != 'cuda':
            raise ValueError(
                'The MMDetection distributed launcher requires CUDA.')
        init_dist(args.launcher, **cfg.dist_params)
    rank, world = get_dist_info()
    cfg.gpu_ids = list(range(world)) if distributed else [args.gpu_id]
    if cfg.device == 'cuda':
        torch.cuda.set_device(
            int(os.environ['LOCAL_RANK']) if distributed else args.gpu_id)
    # Resolve the base seed on resume before constructing data loaders or modules.
    seed = args.seed if args.seed is not None else cfg.get('seed')
    if seed is None and cfg.resume_from:
        seed = torch.load(cfg.resume_from,
                          map_location='cpu').get('meta', {}).get('seed')
    cfg.seed = init_random_seed(seed, device=cfg.device)
    cfg.diff_seed = args.diff_seed or cfg.get('diff_seed', False)
    set_random_seed(cfg.seed + (rank if cfg.diff_seed else 0),
                    deterministic=args.deterministic)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    logger = get_root_logger(log_file=str(
        Path(cfg.work_dir) / f'{timestamp}.log'),
                             log_level=cfg.log_level)
    dataset = build_dataset(cfg.data.train)
    if not len(dataset):
        raise ValueError('The filtered training dataset is empty.')
    classes = prepare_distillation(cfg, dataset)
    if not args.no_validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        if validate_class_order(val_dataset) != classes:
            raise ValueError('Training and validation class order differ.')
    protocol = training_protocol(cfg, dataset, classes, world)
    model_cfg = without_pretraining(
        cfg.model) if cfg.resume_from or cfg.load_from else cfg.model
    model = build_detector(ConfigDict(model_cfg))
    model.init_weights()
    model.CLASSES = classes
    resume_metadata = validate_checkpoint(model, cfg, classes, protocol)
    # IterBasedRunner.resume() does not restore metadata, including AMP state.
    metadata = dict(resume_metadata)
    metadata.update(CLASSES=classes,
                    config=cfg.pretty_text,
                    seed=cfg.seed,
                    irb_protocol=protocol,
                    env_info=collect_env())
    if rank == 0:
        cfg.dump(str(Path(cfg.work_dir) / 'resolved_config.py'))
        (Path(cfg.work_dir) / 'protocol.json').write_text(json.dumps(protocol,
                                                                     indent=2),
                                                          encoding='utf-8')
        logger.info('Environment: %s', metadata['env_info'])
        logger.info('Images=%d, total batch=%d, runner=%s', len(dataset),
                    cfg.data.train_dataloader.samples_per_gpu * world,
                    dict(cfg.runner))
    if cfg.get('checkpoint_config') is not None:
        cfg.checkpoint_config.meta = metadata
    train_detector(model,
                   dataset,
                   cfg,
                   distributed=distributed,
                   validate=not args.no_validate,
                   timestamp=timestamp,
                   meta=metadata)


def main(argv=None):
    args = parse_args(argv)
    train(Config.fromfile(args.config), args)


if __name__ == '__main__':
    main()
