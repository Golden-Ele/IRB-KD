"""Configuration, GT conversion and RoI-level response distillation."""

from copy import deepcopy
from pathlib import Path
import math

import torch
from torch import nn
import torch.distributed as dist
import yaml

from .boundary import BoundaryLoss, encode_distances, valid_distances
from .irb_loss import InteractionLoss, ReshufflingLoss, _fp32_context


DEFAULTS = dict(
    feature_distiller='irb-feature', logic_distiller='irb-logit',
    feature_layers=['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4'],
    channels_s=[256, 512, 1024, 2048], channels_t=[256, 512, 1024, 2048],
    alpha=4e-5, beta=0.8, gamma=6e-5, relation_temperature=0.5,
    boundary_temperature=0.4, boundary_criterion='w1', rho_top=0.2, rho_tail=0.2,
    roi_size=7, embed_dim=128, schedule='constant', class_counts=None)


def load_irb_config(source=None):
    values = source
    if isinstance(source, (str, Path)):
        with open(source, encoding='utf-8') as file:
            values = yaml.safe_load(file)
    if values is not None and not isinstance(values, dict):
        raise ValueError('Distillation configuration must be a mapping.')
    cfg = deepcopy(DEFAULTS)
    unknown = set(values or {}) - set(cfg)
    if unknown:
        raise ValueError(f'Unknown distillation options: {sorted(unknown)}.')
    cfg.update(values or {})
    if cfg['feature_distiller'] not in (None, 'irb-feature'):
        raise ValueError('Use irb-feature or None for feature_distiller.')
    if cfg['logic_distiller'] not in (None, 'irb-logit'):
        raise ValueError('Use irb-logit or None for logic_distiller.')
    if cfg['feature_distiller'] is None and cfg['logic_distiller'] is None:
        raise ValueError('Use baseline training when both distillation branches are disabled.')
    if cfg['schedule'] != 'constant':
        raise ValueError('The configured IRB objective uses constant loss weights.')
    for key in ('alpha', 'beta', 'gamma'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    if cfg['feature_distiller'] is not None and not (
            len(cfg['feature_layers']) == len(cfg['channels_s']) == len(cfg['channels_t']) > 0):
        raise ValueError('Feature layers and channel lists must have equal, nonzero lengths.')
    if cfg['feature_distiller'] == 'irb-feature' and cfg['alpha'] == 0:
        cfg['feature_distiller'] = None
    if cfg['logic_distiller'] == 'irb-logit' and cfg['beta'] == 0 and cfg['gamma'] == 0:
        cfg['logic_distiller'] = None
    if cfg['feature_distiller'] is None and cfg['logic_distiller'] is None:
        raise ValueError('All distillation branches are disabled; use baseline training.')
    return cfg


def count_training_classes(dataset):
    """Count the same non-ignored objects and class IDs exposed by the training dataset."""
    counts = [0] * len(dataset.CLASSES)
    for index in range(len(dataset)):
        for label in dataset.get_ann_info(index)['labels']:
            counts[int(label)] += 1
    return counts


def interaction_batch(images, gt_bboxes):
    h, w = images.shape[-2:]
    boxes, indices = [], []
    for index, xyxy in enumerate(gt_bboxes):
        xyxy = xyxy.detach().float()
        xywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), 1)
        boxes.append(xywh / xywh.new_tensor([w, h, w, h]))
        indices.append(xyxy.new_full((len(xyxy),), index))
    return dict(img=images, bboxes=torch.cat(boxes), batch_idx=torch.cat(indices))


class IRBLogitLoss(nn.Module):
    def __init__(self, support, config):
        super().__init__()
        self.config = config
        if config['beta'] > 0:
            if config['class_counts'] is None:
                raise ValueError('Reshuffling requires training-set class counts.')
            self.reshuffling = ReshufflingLoss(config['class_counts'], config['rho_top'], config['rho_tail'])
        self.boundary = BoundaryLoss(support, config['boundary_temperature'], config['boundary_criterion'])
        self.last_components = {}

    def forward(self, student, teacher, student_head, teacher_head):
        labels = student['labels']
        positive = (labels >= 0) & (labels < student_head.num_classes)
        loss = student['cls_score'].float().sum() * 0 + student['bbox_pred'].float().sum() * 0
        self.last_components = {}
        if self.config['beta'] > 0:
            reshuffling = self.reshuffling(student['cls_score'][positive, :-1],
                                           teacher['cls_score'][positive, :-1])
            loss = loss + self.config['beta'] * reshuffling
            self.last_components['reshuffling'] = reshuffling.detach()
        if self.config['gamma'] > 0:
            distances = encode_distances(student['rois'][positive, 1:], student['targets'][positive])
            valid = valid_distances(distances, student_head.support)
            selected_s = student_head.select_class(student['bbox_pred'][positive], labels[positive])[valid]
            selected_t = teacher_head.select_class(teacher['bbox_pred'][positive], labels[positive])[valid]
            boundary = self.boundary(selected_s, selected_t)
            loss = loss + self.config['gamma'] * boundary
            self.last_components['boundary'] = boundary.detach()
        return loss


class DistillationLoss(nn.Module):
    load_config = staticmethod(load_irb_config)

    def __init__(self, student, teacher, config):
        super().__init__()
        self.config = cfg = load_irb_config(config)
        self.feature_distiller, self.logic_distiller = cfg[
            'feature_distiller'], cfg['logic_distiller']
        object.__setattr__(self, '_student', student)
        object.__setattr__(self, '_teacher', teacher)
        self.origin_outputs, self.teacher_outputs, self.remove_handle = {}, {}, []
        self.active = False
        self.last_components = {}
        if self.feature_distiller == 'irb-feature':
            self.D_loss_fn = InteractionLoss(cfg['channels_s'],
                                             cfg['channels_t'],
                                             cfg['roi_size'], cfg['embed_dim'],
                                             cfg['relation_temperature'])
        if self.logic_distiller == 'irb-logit':
            self.D_loss_lg = IRBLogitLoss(student.roi_head.bbox_head.support,
                                          cfg)
        self.register_hook()

    def register_hook(self):
        if self.remove_handle:
            return

        def hook(cache, key, part, detach):

            def capture(module, inputs, output):
                if self.active:
                    value = output if part is None else output[part]
                    cache[key] = value.detach() if detach else value

            return capture

        if self.feature_distiller:
            for model, cache, detach in ((self._student, self.origin_outputs,
                                          False),
                                         (self._teacher, self.teacher_outputs,
                                          True)):
                modules = dict(model.named_modules())
                for key, name in enumerate(self.config['feature_layers']):
                    path, sep, part = name.partition(':')
                    if path not in modules:
                        raise ValueError(f'Feature layer not found: {path}')
                    self.remove_handle.append(
                        modules[path].register_forward_hook(
                            hook(cache, key,
                                 int(part) if sep else None, detach)))

    def clear_outputs(self):
        self.origin_outputs.clear()
        self.teacher_outputs.clear()

    def remove_handle_(self):
        for handle in self.remove_handle:
            handle.remove()
        self.remove_handle.clear()
        self.clear_outputs()

    def teacher_input(self, img):
        return img

    def forward(self, images, gt_bboxes, student=None, teacher=None):
        with _fp32_context(images.device):
            batch = interaction_batch(images, gt_bboxes)
            loss = images.new_zeros((), dtype=torch.float32)
            self.last_components = {}
            if self.feature_distiller:
                n = len(self.config['feature_layers'])
                if len(self.origin_outputs) != n or len(
                        self.teacher_outputs) != n:
                    raise RuntimeError(
                        'Both detectors must run with feature hooks enabled.')
                ys, yt = [self.origin_outputs[i] for i in range(n)
                          ], [self.teacher_outputs[i] for i in range(n)]
                for i, (s, t) in enumerate(zip(ys, yt)):
                    if s.shape[1] != self.config['channels_s'][i] or t.shape[
                            1] != self.config['channels_t'][i]:
                        raise ValueError(
                            'Feature channels differ from the hook configuration.'
                        )
                feature = self.D_loss_fn(ys, yt, batch)
                loss = loss + self.config['alpha'] * feature
                self.last_components['feature'] = feature.detach()
            if self.logic_distiller:
                sh, th = self._student.roi_head.bbox_head, self._teacher.roi_head.bbox_head
                response = self.D_loss_lg(student, teacher, sh, th)
                self.last_components.update(self.D_loss_lg.last_components)
                loss = loss + response
            # These losses return local contributions to a global objective; DDP averages gradients.
            world_size = dist.get_world_size(
            ) if dist.is_available() and dist.is_initialized() else 1
            return loss * world_size
