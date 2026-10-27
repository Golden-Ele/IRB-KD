"""IRB-KD configuration and teacher-based foreground assignment for YOLOv8."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from .irb_loss import BoundaryLoss, ReshufflingLoss, _fp32_context


def load_irb_config(config=None):
    """Load distillation settings separately from detector hyperparameters."""
    defaults = {
        'feature_distiller': 'irb-feature', 'logic_distiller': 'irb-logit',
        'feature_layers': ['model.2.cv2', 'model.4.cv2', 'model.6.cv2', 'model.8.cv2'],
        'alpha': 4e-5, 'beta': 0.8, 'gamma': 6e-5,
        'relation_temperature': 0.5, 'boundary_temperature': 0.4,
        'boundary_criterion': 'w1',
        'rho_top': 0.2, 'rho_tail': 0.2, 'roi_size': 7, 'embed_dim': 128,
        'class_counts': None, 'topk': 13, 'schedule': 'constant',
    }
    if config is None:
        return defaults
    if isinstance(config, (str, Path)):
        with open(config, encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('distill_cfg must be a YAML mapping or a dictionary.')
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown distillation options: {sorted(unknown)}')
    defaults.update(config)
    for name in ('alpha', 'beta', 'gamma'):
        if not isinstance(defaults[name], (int, float)) or not np.isfinite(defaults[name]) or defaults[name] < 0:
            raise ValueError(f'{name} must be a finite nonnegative loss weight.')
    if defaults['schedule'] not in ('constant', 'cosine'):
        raise ValueError('KD schedule must be constant or cosine.')
    if defaults['boundary_criterion'] not in ('w1', 'kl'):
        raise ValueError('Boundary criterion must be w1 or kl.')
    if not isinstance(defaults['topk'], int) or defaults['topk'] < 1:
        raise ValueError('topk must be a positive integer.')
    return defaults


def count_training_classes(dataset, nc):
    """Count classes in the unaugmented training annotations."""
    labels = getattr(dataset, 'labels', None)
    if labels is None:
        raise ValueError('Automatic IRB class counts require a dataset.labels annotation list.')
    counts = np.zeros(nc, dtype=np.int64)
    for label in labels:
        classes = np.asarray(label['cls'] if isinstance(label, dict) else label[:, 0]).reshape(-1)
        if not np.all(np.isfinite(classes)) or np.any(classes != classes.astype(np.int64)):
            raise ValueError('Training class IDs must be finite integers.')
        classes = classes.astype(np.int64)
        if np.any((classes < 0) | (classes >= nc)):
            raise ValueError(f'Training class IDs must be between 0 and {nc - 1}.')
        counts += np.bincount(classes, minlength=nc)
    return counts.tolist()


class IRBLogitLoss(nn.Module):
    """Reshuffling and boundary distillation on teacher-assigned positives."""

    def __init__(self, student_head, teacher_head, class_counts, config):
        super().__init__()
        for attr in ('nc', 'nl', 'reg_max'):
            if getattr(student_head, attr) != getattr(teacher_head, attr):
                raise ValueError(f'IRB requires matching teacher/student Detect.{attr}.')
        if student_head.reg_max < 2:
            raise ValueError('IRB boundary distillation requires native DFL distributions (reg_max > 1).')
        if not torch.equal(student_head.stride.cpu(), teacher_head.stride.cpu()):
            raise ValueError('IRB requires matching teacher/student detection strides and DFL support.')
        self.nc, self.reg_max, self.nl = student_head.nc, student_head.reg_max, student_head.nl
        self.no = self.nc + 4 * self.reg_max
        self.register_buffer('stride', student_head.stride.detach().clone().float())
        self.register_buffer('proj', torch.arange(self.reg_max, dtype=torch.float32))
        self.beta, self.gamma = config['beta'], config['gamma']
        if self.beta:
            if class_counts is None or len(class_counts) != self.nc:
                raise ValueError(f'IRB reshuffling needs {self.nc} training class counts in class ID order.')
            self.reshuffling = ReshufflingLoss(class_counts, config['rho_top'], config['rho_tail'])
        self.boundary = BoundaryLoss(config['boundary_temperature'], config.get('boundary_criterion', 'w1'))
        self.assigner = TaskAlignedAssigner(topk=config['topk'], num_classes=self.nc, alpha=0.5, beta=6.0)
        self.last_components = {}

    def _unpack(self, predictions):
        feats = predictions[1] if isinstance(predictions, tuple) else predictions
        if not isinstance(feats, (list, tuple)) or len(feats) != self.nl:
            raise ValueError('IRB needs raw YOLOv8 Detect outputs for every detection level.')
        if any(f.ndim != 4 or f.shape[1] != self.no for f in feats):
            raise ValueError('IRB received incompatible raw detection map dimensions.')
        flat = torch.cat([f.reshape(f.shape[0], self.no, -1) for f in feats], dim=2).transpose(1, 2)
        return feats, flat[..., :4 * self.reg_max], flat[..., 4 * self.reg_max:]

    @torch.no_grad()
    def _foreground(self, teacher_features, teacher_dist, teacher_scores, batch):
        device = teacher_scores.device
        batch_size = teacher_scores.shape[0]
        indices = batch['batch_idx'].to(device).reshape(-1).long()
        labels = batch['cls'].to(device).reshape(-1, 1)
        boxes = batch['bboxes'].to(device).float()
        # PyTorch 2.0 deterministic mode requires CPU bincount.
        counts = torch.bincount(indices.cpu(), minlength=batch_size)
        targets = teacher_scores.new_zeros((batch_size, int(counts.max()), 5), dtype=torch.float32)
        height, width = batch['img'].shape[-2:]
        boxes = xywh2xyxy(boxes * boxes.new_tensor([width, height, width, height]))
        for i in range(batch_size):
            mask = indices == i
            n = int(counts[i])
            if n:
                targets[i, :n] = torch.cat((labels[mask], boxes[mask]), dim=1)
        gt_labels, gt_boxes = targets.split((1, 4), dim=2)
        valid = gt_boxes.sum(dim=-1, keepdim=True) > 0
        anchors, strides = make_anchors(teacher_features, self.stride, 0.5)
        distances = teacher_dist.float().reshape(batch_size, -1, 4, self.reg_max).softmax(-1).matmul(self.proj)
        decoded = dist2bbox(distances, anchors, xywh=False) * strides
        _, _, _, foreground, _ = self.assigner(teacher_scores.float().sigmoid(), decoded,
                                               anchors * strides, gt_labels, gt_boxes, valid)
        return foreground.bool()

    def forward(self, s_predictions, t_predictions, batch):
        s_features, s_dist, s_scores = self._unpack(s_predictions)
        t_features, t_dist, t_scores = self._unpack(t_predictions)
        if any(s.shape != t.shape for s, t in zip(s_features, t_features)):
            raise ValueError('IRB teacher/student raw detection grids must match.')
        # All probability/CDF calculations are in FP32, including under AMP.
        with _fp32_context(s_scores.device):
            foreground = self._foreground(t_features, t_dist.detach(), t_scores.detach(), batch)
            res = s_scores.float().sum() * 0
            bou = s_dist.float().sum() * 0
            if self.beta:
                res = self.reshuffling(s_scores[foreground].float(), t_scores[foreground].detach().float())
            if self.gamma:
                bou = self.boundary(s_dist[foreground].float().reshape(-1, 4, self.reg_max),
                                    t_dist[foreground].detach().float().reshape(-1, 4, self.reg_max))
        self.last_components = {'reshuffling': res.detach(), 'boundary': bou.detach()}
        return self.beta * res + self.gamma * bou
