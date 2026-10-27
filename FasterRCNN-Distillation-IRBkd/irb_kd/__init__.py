"""Register the detector, heads and training hooks with MMDetection."""

from .distribution_bbox_head import DistributionShared2FCBBoxHead
from .roi_head import DistillationRoIHead
from .faster_rcnn import IRBFasterRCNN
from .optim import FiniteOptimizerHook, IterationStateHook, TrainableOptimizerConstructor

__all__ = [
    'DistributionShared2FCBBoxHead', 'DistillationRoIHead', 'IRBFasterRCNN',
    'FiniteOptimizerHook', 'IterationStateHook', 'TrainableOptimizerConstructor',
]
