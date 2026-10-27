"""Feature, classification and DFL boundary distillation losses."""

from contextlib import nullcontext
import math

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.ops import roi_align

__all__ = ("InteractionLoss", "ReshufflingLoss", "BoundaryLoss")


def _fp32_context(device):
    """Disable AMP, including CPU AMP, while retaining support for older PyTorch."""
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, enabled=False)
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _paired_logits(student, teacher, ndim):
    if student.ndim != ndim or teacher.shape != student.shape:
        raise ValueError("Student and teacher logits must have the same {}D shape.".format(ndim))
    if teacher.device != student.device:
        raise ValueError("Student and teacher logits must be on the same device.")


def _kl_from_logs(log_student, log_teacher):
    """Elementwise KL contributions; equal log distributions produce exact zero."""
    return log_teacher.exp() * (log_teacher - log_student)


def _global_count(local_count, device):
    """Sum a normalization count across ranks, including ranks with no foreground."""
    count = torch.tensor(local_count, dtype=torch.long, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return count


def _global_logsumexp(local_log_sum):
    """Reduce group log densities across ranks, including empty ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return local_log_sum
    maximum = local_log_sum.clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    finite_maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    shifted_sum = (local_log_sum - finite_maximum).exp()
    dist.all_reduce(shifted_sum, op=dist.ReduceOp.SUM)
    return finite_maximum + shifted_sum.log()


class InteractionLoss(nn.Module):
    """Distill foreground features and relations within and across feature levels.

    Channel lists describe matching levels in shallow-to-deep order. Shared
    projections produce embeddings of width ``embed_dim * roi_size ** 2``;
    only the student branch updates their parameters.
    """

    def __init__(self, channels_s, channels_t, roi_size=7, embed_dim=128,
                 relation_temperature=0.5):
        super().__init__()
        self.channels_s = tuple(int(c) for c in channels_s)
        self.channels_t = tuple(int(c) for c in channels_t)
        if not self.channels_s or len(self.channels_s) != len(self.channels_t):
            raise ValueError("channels_s and channels_t need the same nonzero number of levels.")
        if min(self.channels_s + self.channels_t) <= 0:
            raise ValueError("All feature channel counts must be positive.")
        if isinstance(roi_size, bool) or int(roi_size) != roi_size or roi_size <= 0:
            raise ValueError("roi_size must be a positive integer.")
        if isinstance(embed_dim, bool) or int(embed_dim) != embed_dim or embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")
        if not math.isfinite(relation_temperature) or relation_temperature <= 0:
            raise ValueError("relation_temperature must be finite and positive.")
        self.roi_size = int(roi_size)
        self.embed_dim = int(embed_dim)
        self.relation_temperature = float(relation_temperature)
        self.align_module = nn.ModuleList(
            nn.Identity() if cs == ct else nn.Conv2d(cs, ct, kernel_size=1)
            for cs, ct in zip(self.channels_s, self.channels_t)
        )
        self.projection_heads = nn.ModuleList(
            nn.Conv2d(ct, self.embed_dim, kernel_size=1, bias=False) for ct in self.channels_t
        )
        self.last_components = {}

    def _validate_features(self, y_s, y_t):
        if len(y_s) != len(self.channels_s) or len(y_t) != len(self.channels_t):
            raise ValueError("The number of captured feature levels does not match the configured channels.")
        batch_size, device = y_s[0].shape[0], y_s[0].device
        for level, (student, teacher, cs, ct) in enumerate(
                zip(y_s, y_t, self.channels_s, self.channels_t)):
            if student.ndim != 4 or teacher.ndim != 4:
                raise ValueError("Interaction features at level {} must be BCHW tensors.".format(level))
            if student.shape[1] != cs or teacher.shape[1] != ct:
                raise ValueError("Captured feature channels at level {} differ from the configuration.".format(level))
            if student.shape[0] != batch_size or teacher.shape[0] != batch_size:
                raise ValueError("All feature levels must have the same batch size.")
            if student.shape[-2:] != teacher.shape[-2:]:
                raise ValueError("Student and teacher feature resolutions must match at each selected level.")
            if student.device != device or teacher.device != device:
                raise ValueError("All student and teacher features must be on the same device.")
        if batch_size == 0:
            raise ValueError("Interaction distillation requires a nonempty image batch.")
        return batch_size, device

    @staticmethod
    def _boxes(batch, batch_size, device):
        """Convert normalized xywh labels into clipped normalized xyxy boxes."""
        if "img" not in batch or "bboxes" not in batch or "batch_idx" not in batch:
            raise ValueError("InteractionLoss needs batch['img'], batch['bboxes'] and batch['batch_idx'].")
        image_shape = batch["img"].shape
        if len(image_shape) != 4 or image_shape[0] != batch_size:
            raise ValueError("batch['img'] must be BCHW and match the captured feature batch.")
        image_h, image_w = image_shape[-2:]
        if image_h <= 0 or image_w <= 0:
            raise ValueError("Training images must have positive spatial dimensions.")
        boxes = torch.as_tensor(batch["bboxes"], device=device, dtype=torch.float32).detach()
        indices = torch.as_tensor(batch["batch_idx"], device=device).detach().reshape(-1)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or len(indices) != len(boxes):
            raise ValueError("GT bboxes must be Nx4 normalized xywh with N matching batch indices.")
        if not torch.isfinite(boxes).all() or not torch.isfinite(indices).all():
            raise ValueError("GT boxes and batch indices must be finite.")
        integer_indices = indices.long()
        if ((indices != integer_indices) | (integer_indices < 0) | (integer_indices >= batch_size)).any():
            raise ValueError("GT batch indices must be integers in [0, batch_size).")
        xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2), dim=1).clamp(0, 1)
        size = xyxy[:, 2:] - xyxy[:, :2]
        areas = size[:, 0] * image_w * size[:, 1] * image_h
        valid = (boxes[:, 2:] > 0).all(dim=1) & (size > 0).all(dim=1) & (areas > 0)
        return xyxy[valid], integer_indices[valid], areas[valid]

    @staticmethod
    def _foreground_map(boxes, areas, height, width):
        """Weight inclusive box regions by inverse pixel area; use max at overlaps."""
        weights = boxes.new_zeros((height, width))
        scale = boxes.new_tensor((width, height, width, height))
        projected = boxes * scale
        lower = projected[:, :2].floor().long()
        upper = projected[:, 2:].ceil().long()
        lower[:, 0].clamp_(0, width - 1)
        lower[:, 1].clamp_(0, height - 1)
        upper[:, 0].clamp_(0, width - 1)
        upper[:, 1].clamp_(0, height - 1)
        bounds = torch.cat((lower, upper), dim=1).tolist()
        inverse_areas = areas.reciprocal()
        for k, (x1, y1, x2, y2) in enumerate(bounds):
            region = weights[y1:y2 + 1, x1:x2 + 1]
            weights[y1:y2 + 1, x1:x2 + 1] = torch.maximum(region, inverse_areas[k])
        return weights

    def forward(self, y_s, y_t, batch):
        """Average per-image relations globally; empty images contribute zero."""
        batch_size, device = self._validate_features(y_s, y_t)
        with _fp32_context(device):
            # DDP requires gradients for all heads, including empty-GT batches.
            zero = sum(feature.float().sum() * 0 for feature in y_s)
            zero = zero + sum(parameter.float().sum() * 0 for parameter in self.parameters())
            global_batch_size = _global_count(batch_size, device)
            boxes, indices, areas = self._boxes(batch, batch_size, device)
            if not len(boxes):
                self.last_components = {name: zero.detach() for name in ("ifrd", "frcd", "n2n", "e2e", "clcd")}
                return zero

            students = [align(student.float()) for align, student in zip(self.align_module, y_s)]
            teachers = [teacher.detach().float() for teacher in y_t]
            levels = len(students)
            ifrd, frcd, n2n, e2e = zero, zero, zero, zero
            for image_index in range(batch_size):
                selected = indices == image_index
                if not selected.any():
                    continue
                image_boxes, image_areas = boxes[selected], areas[selected]
                relation_s, relation_t, projected_s, projected_t = [], [], [], []
                for level, (student, teacher) in enumerate(zip(students, teachers)):
                    height, width = teacher.shape[-2:]
                    weights = self._foreground_map(image_boxes, image_areas, height, width)
                    squared_error = (student[image_index] - teacher[image_index]).square()
                    ifrd = ifrd + (squared_error * weights.unsqueeze(0)).sum() / (
                        teacher.shape[1] * weights.sum()) / (levels * global_batch_size)

                    roi_boxes = image_boxes * image_boxes.new_tensor((width, height, width, height))
                    # RoIAlign requires matching FP32 features and coordinates.
                    roi_s = roi_align(student[image_index:image_index + 1], [roi_boxes],
                                      self.roi_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False)
                    with torch.no_grad():
                        roi_t = roi_align(teacher[image_index:image_index + 1], [roi_boxes],
                                          self.roi_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False)
                    nodes_s = F.normalize(roi_s.flatten(1), dim=1)
                    nodes_t = F.normalize(roi_t.flatten(1), dim=1)
                    log_s = F.log_softmax(nodes_s @ nodes_s.t() / self.relation_temperature, dim=1)
                    log_t = F.log_softmax(nodes_t @ nodes_t.t() / self.relation_temperature, dim=1)
                    frcd = frcd + _kl_from_logs(log_s, log_t).sum() / (
                        len(image_boxes) * levels * global_batch_size)
                    relation_s.append(log_s.exp())
                    relation_t.append(log_t.exp())
                    if levels > 1:
                        projected_s.append(self.projection_heads[level](roi_s).flatten(1))
                        with torch.no_grad():
                            projected_t.append(self.projection_heads[level](roi_t).flatten(1))

                for level in range(levels - 1):
                    node_s = F.cosine_similarity(projected_s[level], projected_s[level + 1], dim=1)
                    node_t = F.cosine_similarity(projected_t[level], projected_t[level + 1], dim=1)
                    edge_s = F.cosine_similarity(relation_s[level], relation_s[level + 1], dim=1)
                    edge_t = F.cosine_similarity(relation_t[level], relation_t[level + 1], dim=1)
                    n2n = n2n + (node_s - node_t).square().mean() / ((levels - 1) * global_batch_size)
                    e2e = e2e + (edge_s - edge_t).square().mean() / ((levels - 1) * global_batch_size)

            clcd = n2n + e2e
            self.last_components = {"ifrd": ifrd.detach(), "frcd": frcd.detach(),
                                    "n2n": n2n.detach(), "e2e": e2e.detach(), "clcd": clcd.detach()}
            return ifrd + frcd + clcd


class ReshufflingLoss(nn.Module):
    """Calibrate frequency-ranked groups and sum group and within-group KL.

    Class counts follow detector class-ID order; ties use ascending class ID.
    Teacher densities span all DDP ranks. Each rank returns its local KL sum,
    without foreground normalization or a temperature multiplier.
    """

    def __init__(self, class_counts, rho_top=0.2, rho_tail=0.2):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float64)
        if counts.ndim != 1 or not counts.numel():
            raise ValueError("class_counts must be a nonempty sequence in detector class-ID order.")
        if not torch.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
            raise ValueError("class_counts must be finite, nonnegative, with at least one positive count.")
        if (not math.isfinite(rho_top) or not math.isfinite(rho_tail) or
                rho_top < 0 or rho_tail < 0 or rho_top + rho_tail >= 1):
            raise ValueError("rho_top/rho_tail must be nonnegative and sum to less than one.")
        self.rho_top, self.rho_tail = float(rho_top), float(rho_tail)
        self.register_buffer("class_counts", counts.float())
        class_order = sorted(range(len(counts)), key=lambda j: (-float(counts[j]), j))
        centered_ranks = torch.empty(len(counts), dtype=torch.float64)
        for rank, class_id in enumerate(class_order):
            centered_ranks[class_id] = (rank + 0.5) / len(counts)
        top = centered_ranks <= self.rho_top
        tail = centered_ranks >= 1 - self.rho_tail
        middle = ~(top | tail)
        # Older Gloo backends cannot broadcast boolean module buffers.
        self.register_buffer("group_masks", torch.stack((top, middle, tail)).to(torch.uint8))
        self.last_components = {}

    def forward(self, s_logits, t_logits):
        """Compare paired NxC logits at the same foreground anchors."""
        _paired_logits(s_logits, t_logits, 2)
        if s_logits.shape[1] != len(self.class_counts):
            raise ValueError("Classification logits do not match the number of training class counts.")
        with _fp32_context(s_logits.device):
            student, teacher = s_logits.float(), t_logits.detach().float()
            zero = student.sum() * 0
            masks = self.group_masks.to(device=student.device, dtype=torch.bool)
            active = [mask for mask in masks if mask.any()]
            group_logits_s = torch.stack([torch.logsumexp(student[:, mask], dim=1) for mask in active], dim=1)
            group_logits_t = torch.stack([torch.logsumexp(teacher[:, mask], dim=1) for mask in active], dim=1)
            group_log_s = F.log_softmax(group_logits_s, dim=1)
            group_log_t = F.log_softmax(group_logits_t, dim=1)

            # Log-space calibration avoids underflow; omit empty class groups.
            log_density = _global_logsumexp(torch.logsumexp(group_log_t, dim=0, keepdim=True))
            if not student.shape[0]:
                self.last_components = {"between": zero.detach(), "in_group": zero.detach()}
                return zero
            calibrated_log_t = F.log_softmax(group_log_t - log_density, dim=1)
            between = _kl_from_logs(group_log_s, calibrated_log_t).sum()
            in_group = zero
            for mask in active:
                if int(mask.sum()) > 1:
                    log_s = F.log_softmax(student[:, mask], dim=1)
                    log_t = F.log_softmax(teacher[:, mask], dim=1)
                    in_group = in_group + _kl_from_logs(log_s, log_t).sum()
            self.last_components = {"between": between.detach(), "in_group": in_group.detach()}
            return between + in_group


class BoundaryLoss(nn.Module):
    """W1 or teacher-to-student KL on Nx4xR logits with unit-spaced bins.

    Sum over bins and edges, then divide by the global foreground count.
    The trainer scales local contributions by world size before DDP averaging.
    KL uses the same temperature and reduction, without a temperature-squared factor.
    """

    def __init__(self, temperature=0.4, criterion='w1'):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Boundary temperature must be finite and positive.")
        if criterion not in ('w1', 'kl'):
            raise ValueError("Boundary criterion must be w1 or kl.")
        self.temperature = float(temperature)
        self.criterion = criterion

    def forward(self, s_logits, t_logits):
        _paired_logits(s_logits, t_logits, 3)
        if s_logits.shape[1] != 4 or s_logits.shape[2] < 1:
            raise ValueError("Boundary logits must have shape (N, 4, reg_max) with reg_max >= 1.")
        with _fp32_context(s_logits.device):
            student, teacher = s_logits.float(), t_logits.detach().float()
            global_count = _global_count(student.shape[0], student.device)
            if not student.shape[0]:
                return student.sum() * 0
            if self.criterion == 'kl':
                log_s = F.log_softmax(student / self.temperature, dim=-1)
                log_t = F.log_softmax(teacher / self.temperature, dim=-1)
                return _kl_from_logs(log_s, log_t).sum() / global_count
            prob_s = F.softmax(student / self.temperature, dim=-1)
            prob_t = F.softmax(teacher / self.temperature, dim=-1)
            if student.is_cuda and torch.are_deterministic_algorithms_enabled():
                # PyTorch 2.0 forbids CUDA cumsum in deterministic mode. The
                # fixed prefix-sum matrix computes the same CDF over DFL bins.
                bins = student.shape[-1]
                prefix = torch.ones((bins, bins), device=student.device, dtype=student.dtype).triu()
                cdf_s, cdf_t = prob_s.matmul(prefix), prob_t.matmul(prefix)
            else:
                cdf_s, cdf_t = prob_s.cumsum(dim=-1), prob_t.cumsum(dim=-1)
            return (cdf_s - cdf_t).abs().sum() / global_count
