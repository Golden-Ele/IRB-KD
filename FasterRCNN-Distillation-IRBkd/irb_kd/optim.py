"""Optimizer construction, finite-value checks and iteration resume state."""

import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook, IterBasedRunner, OptimizerHook, get_dist_info
from mmcv.runner.optimizer import DefaultOptimizerConstructor
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS


@OPTIMIZER_BUILDERS.register_module()
class TrainableOptimizerConstructor(DefaultOptimizerConstructor):
    """Keep frozen backbone stages and teacher weights out of optimizer groups."""

    def __call__(self, model):
        optimizer = super().__call__(model)
        for group in optimizer.param_groups:
            group['params'] = [parameter for parameter in group['params'] if parameter.requires_grad]
        optimizer.param_groups = [group for group in optimizer.param_groups if group['params']]
        return optimizer


@HOOKS.register_module()
class FiniteOptimizerHook(OptimizerHook):
    """Reject nonfinite loss or gradients before updating model parameters."""

    def _require_finite(self, value, runner, stage):
        flag = value.detach().to(dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not bool(flag):
            rank, _ = get_dist_info()
            record = dict(iter=runner.iter + 1,
                          rank=rank,
                          stage=stage,
                          log_vars=runner.outputs.get('log_vars', {}))
            path = Path(runner.work_dir) / f'nonfinite_rank{rank}.json'
            path.write_text(json.dumps(record, indent=2), encoding='utf-8')
            raise FloatingPointError(
                f'Nonfinite {stage} at iteration {runner.iter + 1}; see {path}'
            )

    def after_train_iter(self, runner):
        loss = runner.outputs['loss']
        self._require_finite(torch.isfinite(loss).all(), runner, 'loss')
        runner.optimizer.zero_grad()
        loss.backward()
        parameters = [
            p for p in runner.model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        valid = torch.stack([
            torch.isfinite(p.grad).all() for p in parameters
        ]).all() if parameters else loss.new_tensor(False, dtype=torch.bool)
        self._require_finite(valid, runner, 'gradient')
        if self.grad_clip is not None:
            norm_type = float(self.grad_clip.get('norm_type', 2.0))
            gradients = [p.grad.detach() for p in parameters]
            if norm_type == math.inf:
                norm = torch.stack([g.abs().max() for g in gradients]).max()
            else:
                norm = torch.linalg.vector_norm(torch.stack([
                    torch.linalg.vector_norm(g, ord=norm_type)
                    for g in gradients
                ]),
                                                ord=norm_type)
            self._require_finite(torch.isfinite(norm), runner, 'gradient_norm')
            factor = (self.grad_clip['max_norm'] /
                      (norm + 1e-6)).clamp(max=1.0)
            for parameter in parameters:
                parameter.grad.mul_(factor)
            runner.log_buffer.update({'grad_norm': float(norm),
                                      'grad_clip_scale': float(factor)},
                                     runner.outputs['num_samples'])
        runner.optimizer.step()


@HOOKS.register_module()
class IterationStateHook(Hook):
    """Restore the next iteration after an after_train_iter checkpoint.

    MMCV 1.7.2 stores the current zero-based index in IterBasedRunner
    checkpoints. Record the number of completed updates separately.
    """

    def before_run(self, runner):
        if isinstance(runner, IterBasedRunner):
            completed = (runner.meta or {}).get('completed_train_iters')
            if completed is not None:
                runner._iter = int(completed)
                runner._inner_iter = int(completed)
                runner.logger.info(
                    'Restored completed training iterations: %d', completed)

    def after_train_iter(self, runner):
        if isinstance(runner, IterBasedRunner):
            if runner.meta is None:
                runner.meta = {}
            runner.meta['completed_train_iters'] = runner.iter + 1
