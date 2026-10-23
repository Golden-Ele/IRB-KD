"""
基于响应的蒸馏框架
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import weighted_loss, weight_reduce_loss

__all__ = ['CrossKDLoss', 'WSLDLoss', 'NKDLoss', 'DKDLoss', 'KDLoss', 'NovelKDLoss', 'KnowledgeDistillationKLDivLoss']


# CrossKD: Cross-Head Knowledge Distillation for Dense Object Detection
def kd_quality_focal_loss(pred, target, weight=None, beta=1, reduction='mean', avg_factor=None):
    num_classes = pred.size(1)
    if weight is not None:
        weight = weight[:, None].repeat(1, num_classes)

    target = target.detach().sigmoid()
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    focal_weight = torch.abs(pred.sigmoid() - target).pow(beta)
    loss = loss * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class CrossKDLoss(nn.Module):

    def __init__(self, use_sigmoid=True, beta=1.0, reduction='mean', loss_weight=1.0):
        super(CrossKDLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss = self.loss_weight * kd_quality_focal_loss(pred, target, weight, beta=self.beta, reduction=reduction,
                                                            avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss


# Weighted soft labels
class WSLDLoss(nn.Module):
    def __init__(self, temp=4.0, alpha=2.25):   # 默认temp=2.0, alpha=2.5
        super(WSLDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, logit_s, logit_t, gt_label):
        # student模型和teacher模型的logit值
        s_input_for_softmax = logit_s / self.temp
        t_input_for_softmax = logit_t / self.temp
        # teacher模型logit值softmax化
        t_soft_label = F.softmax(t_input_for_softmax, dim=1)
        logsoftmax = nn.LogSoftmax()

        softmax_loss = - torch.sum(t_soft_label * logsoftmax(s_input_for_softmax), 1, keepdim=True)

        fc_s_auto = logit_s.detach()
        fc_t_auto = logit_t.detach()
        log_softmax_s = logsoftmax(fc_s_auto)
        log_softmax_t = logsoftmax(fc_t_auto)
        # 真值one-hot编码
        one_hot_label = gt_label
        # student模型预测值与真值的 ce loss
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        # teacher模型预测值与真值的 ce loss
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)
        # 求比值，其中1e-7避免除以一个非常接近零的数，从而减少数值计算中的不稳定性或数值溢出的风险
        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)   # 不能大于1
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss    # L-wsl
        # wsl的loss
        wsld_loss = (self.temp ** 2) * torch.mean(softmax_loss)
        hard_loss = self.hard_loss(fc_s_auto, one_hot_label)

        return self.alpha * wsld_loss + hard_loss


# NKD Normalized Knowledge distillation
class NKDLoss(nn.Module):

    def __init__(self, temp=1.0, gamma=1.5):
        super(NKDLoss, self).__init__()
        self.temp = temp
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logit_s, logit_t, gt_label):
        if len(gt_label.size()) > 1:  # 维度大于1，说明每个样本有多个类别的标签
            label = torch.max(gt_label, dim=1, keepdim=True)[1]  # 找到每个样本中概率最大的类别作为标签，并保持维度
        else:
            label = gt_label.view(len(gt_label), 1)  # 二维数据

        # N*class
        N, c = logit_s.shape
        s_i = self.log_softmax(logit_s)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)   # 从原tensor中获取指定dim和指定index的数据
        t_t = torch.gather(t_i, 1, label).detach()   # 被分离，不会在反向传播中更新

        loss_t = - (t_t * s_t).mean()     # 计算教师模型和学生模型在标签上的交叉熵损失 L-ori

        mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()    # 用于选择logit_s和logit_t中与标签对应的元素
        logit_s = logit_s[mask].reshape(N, -1)
        logit_t = logit_t[mask].reshape(N, -1)

        # N*class
        S_i = self.log_softmax(logit_s / self.temp)
        T_i = F.softmax(logit_t / self.temp, dim=1)

        loss_non = (T_i * S_i).sum(dim=1).mean()    # 非标签部分的损失
        loss_non = - self.gamma * (self.temp ** 2) * loss_non

        return loss_t + loss_non


# DKD Decoupled Knowledge distillation
def _get_gt_mask(logit, gt_label):
    gt_label = gt_label.long()      # 将数字或字符串转换为一个长整型
    # 填充 mask | 在第一个维度上进行填充，gt_label 中的位置上的值设置为1，表示这些位置对应的类别是目标的类别
    mask = torch.zeros_like(logit).scatter_(1, gt_label, 1).bool()
    return mask


def _get_other_mask(logit, gt_label):
    gt_label = gt_label.long()    # 将数字或字符串转换为一个长整型
    mask = torch.ones_like(logit).scatter_(1, gt_label, 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    # 将softmax以后的输出 以及目标与非目标mask进行cat操作
    t1 = (t * mask1).sum(dim=1, keepdims=True)   # 目标类
    t2 = (t * mask2).sum(dim=1, keepdims=True)   # 非目标类
    rt = torch.cat([t1, t2], dim=1)   # # 第一列是目标类别的logit, 第二列是所有非目标类别的logit的和
    return rt


class DKDLoss(nn.Module):

    def __init__(self, temp=4.0, alpha=1.0, beta=6.0):     # 注意T的设定
        super(DKDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.beta = beta

    def forward(self, logit_s, logit_t, gt_label):
        gt_mask = _get_gt_mask(logit_s, gt_label)         # 除了每个样本对应target索引处为True, 其它都为False
        other_mask = _get_other_mask(logit_s, gt_label)
        pred_student = F.softmax(logit_s / self.temp, dim=1)
        pred_teacher = F.softmax(logit_t / self.temp, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)  # 第一列是目标类别的logit, 第二列是所有非目标类别的logit的和
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        # 计算目标类loss
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False) * (self.temp ** 2) / gt_label.shape[0])
        # e^{-1000}非常小约等于0，等价于把这一项去掉了   原本应该是logit[1-gt_mask]/temperature计算所有非目标类别的softmax;
        # 因为这里 index 操作比较慢，因此改成 logit/temperature - 1000 * gt_mask，gt_mask中非目标类别处全为0，因此相当于没减;
        # 目标类别的 logit 减去了1000，相当于softmax中分子和分母各加上e^(−1000),
        pred_teacher_part2 = F.softmax(logit_t / self.temp - 1000.0 * gt_mask, dim=1)
        log_pred_student_part2 = F.log_softmax(logit_s / self.temp - 1000.0 * gt_mask, dim=1)
        # 计算非目标类loss
        nckd_loss = (F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False) * (self.temp ** 2)
                     / gt_label.shape[0])

        return self.alpha * tckd_loss + self.beta * nckd_loss


# Distilling the Knowledge in a Neural Network
class KDLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


# BCKD
@weighted_loss
def novel_kd_loss(pred, soft_label, detach_target=True, beta=1.0):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = soft_label.sigmoid()    # 将logit输出进行sigmoid
    score = pred.sigmoid()

    if detach_target:
        target = target.detach()     # 梯度不需要反向传播

    scale_factor = target - score    # 教师输出和学生输出的差异 即权重
    # _with_logits 表示该损失函数已经内部自带了计算logit的操作，无需在传入给这个loss函数之前手动使用sigmoid/softmax将之前网络的输入映射到[0,1]之间
    kd_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * scale_factor.abs().pow(beta)  # 绝对值
    kd_loss = kd_loss.sum(dim=1, keepdim=False)
    return kd_loss


class NovelKDLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='sum', loss_weight=1.0, T=10, threshold=0.05):
        super(NovelKDLoss, self).__init__()
        # assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T
        self.threshold = threshold

    def forward(self, pred, soft_label, weight=None, avg_factor=None, reduction_override=None, beta=1.0):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * novel_kd_loss(pred, soft_label, weight, reduction=reduction, avg_factor=avg_factor,
                                                   beta=beta)
        return loss_kd


# LD(Localization Distillation for Dense Object Detection)
@weighted_loss
def knowledge_distillation_kl_div_loss(pred, soft_label, T, detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (T * T)

    return kd_loss


class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self, pred, soft_label, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(pred, soft_label, weight, reduction=reduction,
                                                                        avg_factor=avg_factor, T=self.T)

        return loss_kd
