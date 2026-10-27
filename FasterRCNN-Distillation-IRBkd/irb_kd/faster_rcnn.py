"""Faster R-CNN training with a frozen teacher and shared sampled RoIs."""

from copy import deepcopy

import torch
from mmcv import ConfigDict
from mmdet.models import DETECTORS, build_detector
from mmdet.models.detectors import FasterRCNN

from .distribution_bbox_head import DistributionShared2FCBBoxHead
from .roi_head import DistillationRoIHead
from .distillation import DistillationLoss


def without_pretraining(value):
    value = deepcopy(value)
    if isinstance(value, dict):
        if isinstance(value.get('init_cfg'), dict) and value['init_cfg'].get('type') == 'Pretrained':
            value['init_cfg'] = None
        if 'pretrained' in value:
            value['pretrained'] = None
        return {key: without_pretraining(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(without_pretraining(item) for item in value)
    return value


def validate_finite_state(state):
    """Reject corrupted weights before model initialization or resume."""
    bad = [name for name, value in state.items()
           if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex())
           and not torch.isfinite(value).all()]
    if bad:
        raise ValueError(f'Checkpoint contains nonfinite tensors: {bad[:8]}')


def validate_head_checkpoint(model, checkpoint, classes):
    state = checkpoint.get('state_dict', {})
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    validate_finite_state(state)
    head = model.roi_head.bbox_head
    prefix = 'roi_head.bbox_head.'
    support = state.get(prefix + 'support')
    if support is None or support.shape != head.support.shape or not torch.equal(support.cpu(), head.support.cpu()):
        raise ValueError('Checkpoint lacks the configured 17-point boundary support. Train the adapted teacher first.')
    weight = state.get(prefix + 'fc_reg.weight')
    if weight is None or weight.shape != head.fc_reg.weight.shape:
        raise ValueError('Checkpoint is not a compatible learned distributional localization head.')
    if tuple(checkpoint.get('meta', {}).get('CLASSES', ())) != tuple(classes):
        raise ValueError('Checkpoint class names/order do not match the training dataset.')
    return state


@DETECTORS.register_module()
class IRBFasterRCNN(FasterRCNN):
    def __init__(self, teacher_model=None, teacher_checkpoint=None, distillation=None,
                 classes=None, **kwargs):
        super().__init__(**kwargs)
        self.classes = tuple(classes or ())
        self.distillation = None
        if distillation is None:
            if teacher_model is not None or teacher_checkpoint is not None:
                raise ValueError('A teacher requires a distillation configuration.')
            return
        if not teacher_model or not teacher_checkpoint:
            raise ValueError('Distillation requires a teacher model and a trained teacher checkpoint.')
        if not isinstance(self.roi_head, DistillationRoIHead):
            raise ValueError('Use DistillationRoIHead to expose the student sampling results.')
        self.teacher = build_detector(ConfigDict(without_pretraining(teacher_model)))
        sh, th = self.roi_head.bbox_head, self.teacher.roi_head.bbox_head
        if not all(isinstance(head, DistributionShared2FCBBoxHead) for head in (sh, th)):
            raise ValueError('Teacher and student must both use the adapted distributional head.')
        if sh.num_classes != th.num_classes or sh.reg_class_agnostic != th.reg_class_agnostic or not torch.equal(sh.support, th.support):
            raise ValueError('Teacher/student class dimensions, support points and regression mode must match.')
        if len(self.classes) != sh.num_classes:
            raise ValueError('Set classes in dataset class-ID order.')
        checkpoint = torch.load(teacher_checkpoint, map_location='cpu')
        state = validate_head_checkpoint(self.teacher, checkpoint, self.classes)
        self.teacher.load_state_dict(state, strict=True)
        self.teacher.requires_grad_(False).eval()
        self.teacher_checkpoint = str(teacher_checkpoint)
        self.distillation = DistillationLoss(self, self.teacher, distillation)

    def init_weights(self):
        # A loaded teacher must not be initialized again by BaseModule.init_weights().
        teacher = self._modules.pop('teacher', None)
        try:
            super().init_weights()
        finally:
            if teacher is not None:
                self.add_module('teacher', teacher)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'teacher'):
            self.teacher.eval()
        return self

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
        if self.distillation is None:
            return super().forward_train(img, img_metas, gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks, proposals, **kwargs)
        kd = self.distillation
        kd.clear_outputs()
        kd.active = True
        self.roi_head.capture_distillation = kd.logic_distiller is not None
        try:
            losses = super().forward_train(img, img_metas, gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks, proposals, **kwargs)
            student = self.roi_head.distillation_context
            with torch.no_grad():
                teacher_features = self.teacher.extract_feat(kd.teacher_input(img))
                teacher = self.teacher.roi_head._bbox_forward(teacher_features, student['rois']) if student is not None else None
            losses['loss_distillation'] = kd(img, gt_bboxes, student, teacher)
            losses.update({'kd_' + name: value for name, value in kd.last_components.items()})
            return losses
        finally:
            kd.active = False
            kd.clear_outputs()
            self.roi_head.capture_distillation = False
            self.roi_head.distillation_context = None
