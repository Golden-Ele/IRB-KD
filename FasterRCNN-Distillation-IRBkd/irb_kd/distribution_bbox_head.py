"""A learned 17-point localization predictor for the second-stage detector."""

import math

import torch
from torch import nn
from mmcv.runner import force_fp32
from mmdet.core import multiclass_nms
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead

from .boundary import (decode_distances, distribution_focal_loss,
                                   encode_distances, make_support, valid_distances)


@HEADS.register_module()
class DistributionShared2FCBBoxHead(Shared2FCBBoxHead):
    def __init__(self, num_bins=17, d_max=None, dfl_loss_weight=None, **kwargs):
        support = make_support(num_bins, d_max)
        if dfl_loss_weight is None or not math.isfinite(dfl_loss_weight) or dfl_loss_weight <= 0:
            raise ValueError('Set a positive dfl_loss_weight from the training experiment.')
        super().__init__(**kwargs)
        if not self.with_cls or not self.with_reg or self.reg_decoded_bbox:
            raise ValueError('This head requires classification and coder-based box supervision.')
        if self.custom_cls_channels or type(self.loss_bbox).__name__ != 'SmoothL1Loss':
            raise ValueError('Use softmax classification and SmoothL1Loss for this head.')
        self.num_bins = int(num_bins)
        self.reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        self.dfl_loss_weight = float(dfl_loss_weight)
        self.register_buffer('support', support)
        self.fc_reg = nn.Linear(self.reg_last_dim, 4 * self.reg_classes * self.num_bins)

    def distributions(self, bbox_pred):
        return bbox_pred.reshape(len(bbox_pred), self.reg_classes, 4, self.num_bins)

    def select_class(self, bbox_pred, labels):
        distributions = self.distributions(bbox_pred)
        classes = torch.zeros_like(labels) if self.reg_class_agnostic else labels
        return distributions[torch.arange(len(labels), device=labels.device), classes]

    def decode(self, proposals, bbox_pred):
        probabilities = self.distributions(bbox_pred).float().softmax(-1)
        distances = (probabilities * self.support).sum(-1)
        return decode_distances(proposals.float(), distances)

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg):
        labels, label_weights, targets, bbox_weights = super()._get_target_single(
            pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg)
        # Keep exact GT coordinates for the distance targets and encode only in loss().
        targets[:len(pos_gt_bboxes)] = pos_gt_bboxes
        return labels, label_weights, targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights,
             bbox_targets, bbox_weights, reduction_override=None):
        losses = super().loss(cls_score, None, rois, labels, label_weights,
                              bbox_targets, bbox_weights, reduction_override)
        losses.setdefault('loss_cls', cls_score.sum() * 0)
        losses.setdefault('acc', cls_score.new_zeros(()))
        positive = (labels >= 0) & (labels < self.num_classes)
        logits = self.select_class(bbox_pred[positive], labels[positive])
        proposals, targets = rois[positive, 1:].float(), bbox_targets[positive].float()
        if positive.any():
            distances = encode_distances(proposals, targets)
            valid = valid_distances(distances, self.support)
            predicted = decode_distances(proposals, (logits.softmax(-1) * self.support).sum(-1))
            # Clamp only numerical underflow in decoded sizes, not GT distances.
            predicted = torch.cat((predicted[:, :2], torch.maximum(
                predicted[:, 2:], predicted[:, :2] + 1e-6)), dim=-1)
            losses['loss_bbox'] = self.loss_bbox(
                self.bbox_coder.encode(proposals, predicted),
                self.bbox_coder.encode(proposals, targets), bbox_weights[positive],
                avg_factor=max(len(labels), 1), reduction_override=reduction_override)
            losses['loss_dfl'] = self.dfl_loss_weight * distribution_focal_loss(
                logits[valid], distances[valid], self.support).sum() / max(int(valid.sum()), 1)
            losses['dfl_invalid_fraction'] = (~valid).float().mean().detach()
        else:
            losses['loss_bbox'] = bbox_pred.sum() * 0
            losses['loss_dfl'] = bbox_pred.sum() * 0
            losses['dfl_invalid_fraction'] = bbox_pred.new_zeros(())
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor,
                   rescale=False, cfg=None):
        boxes = self.decode(rois[:, 1:], bbox_pred)
        if img_shape is not None:
            limit = boxes.new_tensor([img_shape[1], img_shape[0], img_shape[1], img_shape[0]])
            boxes = torch.minimum(boxes.clamp_min(0), limit)
        if rescale:
            boxes = boxes / boxes.new_tensor(scale_factor)
        boxes = boxes.reshape(len(rois), 4 * self.reg_classes)
        scores = cls_score.float().softmax(-1)
        if cfg is None:
            return boxes, scores
        return multiclass_nms(boxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)

    def regress_by_class(self, *args, **kwargs):
        raise NotImplementedError('Cascade refinement is outside this Faster R-CNN implementation.')

    def onnx_export(self, *args, **kwargs):
        raise NotImplementedError('Use PyTorch inference; a distribution-aware ONNX export is not provided.')
