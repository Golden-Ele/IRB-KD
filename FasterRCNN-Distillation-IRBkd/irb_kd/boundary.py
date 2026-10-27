"""Proposal-normalized boundary geometry and distribution losses."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .irb_loss import _fp32_context, _global_count, _paired_logits


def make_support(num_bins, d_max):
    if num_bins != 17:
        raise ValueError('The Faster R-CNN head requires exactly 17 support points.')
    if d_max is None or not math.isfinite(d_max) or d_max <= 0:
        raise ValueError('Set d_max explicitly to the distance limit used in the experiment.')
    return torch.linspace(0, float(d_max), num_bins)


def proposal_geometry(proposals):
    size = proposals[:, 2:] - proposals[:, :2]
    if not torch.isfinite(proposals).all() or (size <= 0).any():
        raise ValueError('Proposals must have finite coordinates and positive width and height.')
    return (proposals[:, :2] + proposals[:, 2:]) / 2, size


def encode_distances(proposals, boxes):
    center, size = proposal_geometry(proposals)
    return torch.cat(((center - boxes[:, :2]) / size,
                      (boxes[:, 2:] - center) / size), dim=-1)


def decode_distances(proposals, distances):
    """Decode [N, ..., 4] distances using each RoI's center and dimensions."""
    center, size = proposal_geometry(proposals)
    for _ in range(distances.ndim - 2):
        center, size = center.unsqueeze(1), size.unsqueeze(1)
    return torch.cat((center - distances[..., :2] * size,
                      center + distances[..., 2:] * size), dim=-1)


def valid_distances(distances, support):
    return (torch.isfinite(distances).all(-1)
            & (distances >= support[0]).all(-1)
            & (distances <= support[-1]).all(-1))


def distribution_focal_loss(logits, targets, support):
    """Return one interpolated cross entropy per edge, including both endpoints."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError('DFL logits and target dimensions do not match.')
    if not torch.isfinite(targets).all() or ((targets < support[0]) | (targets > support[-1])).any():
        raise ValueError('DFL targets must lie within the support; filter invalid RoIs first.')
    position = (targets - support[0]) / (support[1] - support[0])
    lower = position.floor().long().clamp(0, logits.shape[-1] - 2)
    upper_weight = position - lower.to(position.dtype)
    log_prob = F.log_softmax(logits.float(), dim=-1)
    lo = log_prob.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    hi = log_prob.gather(-1, (lower + 1).unsqueeze(-1)).squeeze(-1)
    return -(1 - upper_weight) * lo - upper_weight * hi


class BoundaryLoss(nn.Module):
    """Sum four edge losses and normalize by the global valid-positive RoI count."""

    def __init__(self, support, temperature=0.4, criterion='w1'):
        super().__init__()
        support = torch.as_tensor(support, dtype=torch.float32)
        if support.ndim != 1 or len(support) < 2 or not torch.isfinite(support).all() or not (support.diff() > 0).all():
            raise ValueError('Support points must be finite and strictly increasing.')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive.')
        if criterion not in ('w1', 'kl'):
            raise ValueError('Boundary criterion must be w1 or kl.')
        self.register_buffer('support', support.clone())
        self.temperature, self.criterion = float(temperature), criterion

    def forward(self, student, teacher):
        _paired_logits(student, teacher, 3)
        if student.shape[1:] != (4, len(self.support)):
            raise ValueError('Boundary logits must have shape [N, 4, K].')
        with _fp32_context(student.device):
            s, t = student.float(), teacher.detach().float()
            count = _global_count(len(s), s.device).clamp_min(1)
            if not len(s):
                return s.sum() * 0
            log_s = F.log_softmax(s / self.temperature, -1)
            log_t = F.log_softmax(t / self.temperature, -1)
            if self.criterion == 'kl':
                return (log_t.exp() * (log_t - log_s)).sum() / count
            # Exclude the last CDF entry and weight each interval by its width.
            prefix = s.new_ones((len(self.support), len(self.support) - 1)).triu()
            cdf_delta = (log_s.exp() - log_t.exp()) @ prefix
            return (cdf_delta.abs() * self.support.diff()).sum() / count
