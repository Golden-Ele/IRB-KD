"""IRB feature and logit distillation hooks."""
import torch
from torch import nn
from .irb_loss import InteractionLoss
from .irb_distillation import IRBLogitLoss, count_training_classes, load_irb_config
__all__ = ['Distillation_loss', 'count_training_classes']

class Distillation_loss(nn.Module):
    """Capture detector outputs and combine feature and logit losses."""
    load_config = staticmethod(load_irb_config)

    def __init__(self, modeln, modelL, feature_distiller='irb-feature', logic_distiller='irb-logit', config=None, class_counts=None):
        super().__init__()
        cfg = self.load_config(config)
        if config is not None:
            feature_distiller, logic_distiller = (cfg['feature_distiller'], cfg['logic_distiller'])
        if feature_distiller not in (None, 'irb-feature') or logic_distiller not in (None, 'irb-logit'):
            raise ValueError('This source snapshot contains only IRB feature and logit distillation.')
        self.feature_distiller, self.logic_distiller = (feature_distiller, logic_distiller)
        self.config, self.schedule = (cfg, cfg['schedule'])
        self.teacher_outputs, self.origin_outputs = ({}, {})
        self.remove_handle = []
        self.student_predictions = self.teacher_predictions = None
        self.last_components = {}
        object.__setattr__(self, '_student', modeln.module if hasattr(modeln, 'module') else modeln)
        object.__setattr__(self, '_teacher', modelL.module if hasattr(modelL, 'module') else modelL)
        self._teacher.requires_grad_(False).eval()
        self.teacher_module_pairs, self.student_module_pairs = ([], [])
        channels_s, channels_t = ([], [])
        if feature_distiller is not None:
            for name in cfg['feature_layers']:
                student_module = self._resolve(self._student, name)
                teacher_module = self._resolve(self._teacher, name)
                self.student_module_pairs.append(student_module)
                self.teacher_module_pairs.append(teacher_module)
                channels_s.append(self._channels(student_module))
                channels_t.append(self._channels(teacher_module))
            if not channels_s:
                raise ValueError('Feature distillation requires at least one feature_layers entry.')
            self.D_loss_fn = InteractionLoss(channels_s, channels_t, roi_size=cfg['roi_size'], embed_dim=cfg['embed_dim'], relation_temperature=cfg['relation_temperature'])
        if logic_distiller is not None:
            student_head, teacher_head = (self._student.model[-1], self._teacher.model[-1])
            self._validate_class_names()
            counts = cfg['class_counts'] if cfg['class_counts'] is not None else class_counts
            self.D_loss_lg = IRBLogitLoss(student_head, teacher_head, counts, cfg)

    @staticmethod
    def _resolve(model, name):
        modules = dict(model.named_modules())
        if name not in modules:
            raise ValueError(f'Feature layer {name!r} is absent. Set feature_layers to existing module names.')
        return modules[name]

    @staticmethod
    def _channels(module):
        conv = module if isinstance(module, nn.Conv2d) else getattr(module, 'conv', None)
        if not isinstance(conv, nn.Conv2d):
            raise ValueError('Feature hooks must select a Conv2d or a Conv wrapper with known output channels.')
        return conv.out_channels

    def _validate_class_names(self):

        def names(model):
            result = getattr(model, 'names', None)
            return [result[i] for i in range(len(result))] if isinstance(result, dict) else result
        student, teacher = (names(self._student), names(self._teacher))
        if student is not None and teacher is not None and (list(student) != list(teacher)):
            raise ValueError('IRB teacher/student class names and class ID order must match the training dataset.')

    def clear_outputs(self):
        self.teacher_outputs.clear()
        self.origin_outputs.clear()
        self.student_predictions = self.teacher_predictions = None

    def teacher_input(self, images):
        """Return the image batch used by the frozen teacher."""
        return images

    def register_hook(self):
        if self.remove_handle:
            return
        self.clear_outputs()

        def feature_hook(cache, index, detach=False):

            def capture(module, inputs, output):
                cache[index] = output.detach() if detach else output
            return capture
        for index, (teacher, student) in enumerate(zip(self.teacher_module_pairs, self.student_module_pairs)):
            self.remove_handle.append(teacher.register_forward_hook(feature_hook(self.teacher_outputs, index, True)))
            self.remove_handle.append(student.register_forward_hook(feature_hook(self.origin_outputs, index)))
        if self.logic_distiller is not None:

            def student_head_hook(module, inputs, output):
                self.student_predictions = output

            def teacher_head_hook(module, inputs, output):
                self.teacher_predictions = output
            self.remove_handle.append(self._student.model[-1].register_forward_hook(student_head_hook))
            self.remove_handle.append(self._teacher.model[-1].register_forward_hook(teacher_head_hook))

    def forward(self, batch):
        loss = batch['img'].new_zeros((), dtype=torch.float32)
        self.last_components = {}
        try:
            if self.feature_distiller is not None:
                n = len(self.student_module_pairs)
                if len(self.origin_outputs) != n or len(self.teacher_outputs) != n:
                    raise RuntimeError('Missing KD features: register hooks before forwarding both detectors.')
                y_s = [self.origin_outputs[i] for i in range(n)]
                y_t = [self.teacher_outputs[i] for i in range(n)]
                feature = self.D_loss_fn(y_s, y_t, batch)
                loss = loss + self.config['alpha'] * feature
                self.last_components.update(self.D_loss_fn.last_components)
                self.last_components['interaction'] = feature.detach()
            if self.logic_distiller is not None:
                if self.student_predictions is None or self.teacher_predictions is None:
                    raise RuntimeError('Missing raw KD predictions: forward both detectors with hooks registered.')
                logic = self.D_loss_lg(self.student_predictions, self.teacher_predictions, batch)
                loss = loss + logic
                self.last_components.update(getattr(self.D_loss_lg, 'last_components', {}))
                self.last_components['logit'] = logic.detach()
            self.last_components['total'] = loss.detach()
            return loss
        finally:
            self.clear_outputs()

    def get_loss(self, batch):
        return self(batch)

    def remove_handle_(self):
        for handle in self.remove_handle:
            handle.remove()
        self.remove_handle.clear()
        self.clear_outputs()
