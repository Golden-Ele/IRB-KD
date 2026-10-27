"""Expose the sampled second-stage RoIs without changing assignment or sampling."""

from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead


@HEADS.register_module()
class DistillationRoIHead(StandardRoIHead):
    capture_distillation = False
    distillation_context = None

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        rois = bbox2roi([result.bboxes for result in sampling_results])
        outputs = self._bbox_forward(x, rois)
        targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        outputs['loss_bbox'] = self.bbox_head.loss(
            outputs['cls_score'], outputs['bbox_pred'], rois, *targets)
        if self.capture_distillation:
            self.distillation_context = dict(
                rois=rois.detach(), labels=targets[0], targets=targets[2],
                cls_score=outputs['cls_score'], bbox_pred=outputs['bbox_pred'])
        return outputs
