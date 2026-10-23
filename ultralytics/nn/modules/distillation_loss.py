"""
选择蒸馏器
"""
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.yolo.utils.metrics import bbox_iou
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox
from ultralytics.nn.modules.kd_loss import NovelKDLoss, KnowledgeDistillationKLDivLoss, KDLoss, DKDLoss, NKDLoss, \
    WSLDLoss, CrossKDLoss
from ultralytics.nn.modules.distillation_build import MGDLoss, CWDLoss, FSPLoss, FGDLoss, PKDLoss, AWDLoss, DMKDLoss

__all__ = ['Distillation_loss', 'FeatureLoss', 'LogicLoss']


class Distillation_loss:
    def __init__(self, modeln, modelL, feature_distiller='irb-feature', logic_distiller='irb-logit'):     # 选择蒸馏器      
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_distiller = feature_distiller
        self.logic_distiller = logic_distiller

        # FeatureLoss
        channels_s = [32, 64, 128, 256]     # 8n对应特征层的通道数-小目标检测层
        # channels_s = [128, 256, 128, 64, 128, 256]     # 8n对应特征层的通道数

        # channels_t = [128, 256, 512, 512]    # 8l对应特征层的通道数-小目标检测层
        # channels_t = [512, 512, 512, 256, 512, 512]    # 8l对应特征层的通道数
        channels_t = [160, 320, 640, 640]  # 8X对应特征层的通道数-小目标检测层

        layers = ["2", "4", "6", "8"]    # 相应特征层的层数-小目标检测层
        # layers = ["6", "8", "12", "15", "18", "21"]    # 相应特征层的层数

        if self.feature_distiller is not None:
            self.D_loss_fn = FeatureLoss(channels_s=channels_s, channels_t=channels_t, distiller=self.feature_distiller)   
        if self.logic_distiller is not None:
            self.model_student = modeln
            self.model_teacher = modelL
            self.D_loss_lg = LogicLoss(self.model_student, self.model_teacher, distiller=self.logic_distiller)

        self.teacher_module_pairs = []
        self.student_module_pairs = []
        self.remove_handle = []
        # 教师模型
        for mname, ml in modelL.named_modules():
            if mname is not None:
                name = mname.split(".")
                if name[0] == "module":
                    name.pop(0)
                if len(name) == 3:
                    if name[1] in layers:
                        if "cv2" in mname:
                            self.teacher_module_pairs.append(ml)
        # 学生模型
        for mname, ml in modeln.named_modules():
            if mname is not None:
                name = mname.split(".")
                if name[0] == "module":
                    name.pop(0)
                if len(name) == 3:
                    if name[1] in layers:
                        if "cv2" in mname:
                            self.student_module_pairs.append(ml)

    def register_hook(self):
        self.teacher_outputs = []
        self.origin_outputs = []

        def make_layer_forward_hook(l):
            def forward_hook(m, input, output):
                l.append(output)

            return forward_hook

        for ml, ori in zip(self.teacher_module_pairs, self.student_module_pairs):
            # 为每层加入钩子，在进行Forward的时候会自动将每层的特征传送给model_outputs和origin_outputs
            self.remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(self.origin_outputs)))

    def get_loss(self, batch):
        quant_loss = torch.zeros(2, device=self.device)       # 1x2的tensor 分别是feature loss和logit loss
        if self.feature_distiller is not None:
            # print('feature_distiller')
            quant_loss[0] += self.D_loss_fn(y_t=self.teacher_outputs, y_s=self.origin_outputs, meta=batch)
            self.teacher_outputs.clear()
            self.origin_outputs.clear()
        if self.logic_distiller is not None:
            pred_s = self.model_student(batch['img'])
            pred_t = self.model_teacher(batch['img'])
            # print('logic_distiller')
            quant_loss[1] += self.D_loss_lg(pred_s, pred_t, batch)     # 逐batch存储蒸馏loss
            self.teacher_outputs.clear()
            self.origin_outputs.clear()
        return quant_loss.sum()

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1, padding=0).to(device)
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]

        if distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        elif distiller == 'fsp':
            self.feature_loss = FSPLoss(channels_s, channels_t)
        elif distiller == 'fgd':
            self.feature_loss = FGDLoss(channels_s, channels_t)
        elif distiller == 'pkd':
            self.feature_loss = PKDLoss(channels_s, channels_t)
        elif distiller == 'awd':
            self.feature_loss = AWDLoss(channels_s, channels_t)
        elif distiller == 'dmkd':
            self.feature_loss = DMKDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t, meta):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)
        if isinstance(self.feature_loss, DMKDLoss):      # 确定哪种蒸馏框架需要gt_bbox信息
            bboxes = torch.cat((meta['batch_idx'].view(-1, 1), meta['bboxes']), 1)    # 添加gt_bbox信息
            loss = self.feature_loss(stu_feats, tea_feats, bboxes)
        else:
            loss = self.feature_loss(stu_feats, tea_feats)
        return self.loss_weight * loss


class LogicLoss(nn.Module):
    def __init__(self, s_model, t_model, loss_weight=1.0, distiller='kd'):
        super(LogicLoss, self).__init__()

        self.loss_weight = loss_weight
        self.distiller = distiller

        device = next(s_model.parameters()).device    # 获取神经网络模型中的第一个参数（通常是一个权重矩阵）
        h = s_model.args         # hyperparameters
        m = s_model.model[-1]    # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc    # number of classes
        self.no = m.no    # self.no = nc + self.reg_max * 4  number of outputs per anchor
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=13, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.grids = [torch.zeros(1)] * m.nl
        self.grid_cell_offset = 0.5
        self.fpn_strides = [8, 16, 32]
        self.grid_cell_size = 5.0
        if self.distiller == 'bckd':
            self.cls_loss_weight = 1.0
            self.box_loss_weight = 4.0
            self.cls_loss = NovelKDLoss(loss_weight=1.0)    # focal loss
            self.box_loss = 'EIoULoss'                      # EIoU-loss
        elif self.distiller == 'ckd':
            self.cls_loss_weight = 1.0
            self.box_loss_weight = 4.0
            self.cls_loss = CrossKDLoss(loss_weight=1.0)
            self.box_loss = KnowledgeDistillationKLDivLoss(loss_weight=1.0)
        elif self.distiller == 'ld':
            self.cls_loss_weight = 1.0
            self.box_loss_weight = 1.0
            self.cls_loss = None
            self.box_loss = KnowledgeDistillationKLDivLoss(loss_weight=1.0)
        elif self.distiller == 'kd':
            self.cls_loss_weight = 1.0
            self.box_loss_weight = 1.0
            self.cls_loss = KDLoss(T=10)
            self.box_loss = None
        elif self.distiller == 'dkd':
            self.cls_loss_weight = 1.0
            self.box_loss_weight = 1.0
            self.cls_loss = DKDLoss()
            self.box_loss = None
        elif self.distiller == 'nkd':
            self.cls_loss_weight = 1.0
            self.box_loss_weight = 1.0
            self.cls_loss = NKDLoss()
            self.box_loss = None
        elif self.distiller == 'wsld':
            self.cls_loss_weight = 1.0
            self.box_loss_weight = 1.0
            self.cls_loss = WSLDLoss()
            self.box_loss = None
        else:
            raise NotImplementedError

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        Preprocesses the target counts and matches with the input batch size to output a tensor
        统计目标信息
        """
        if targets.shape[0] == 0:    # check有无目标
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]        # image index
            # 用于统计每个类别出现的次数
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)

            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)

            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:     # 如果当前存在目标
                    out[j, :n] = targets[matches, 1:]       # 将目标信息
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))    # 坐标信息转换
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution.
           解码预测对象边界框坐标从锚点和分布"""
        if self.use_dfl:
            b, a, c = pred_dist.shape      # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype).to(self.device))    # 将坐标值转换为概率分布
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, s_preds, t_preds, batch):
        loss = torch.zeros(2, device=self.device)      # box loss 和 cls loss
        # 分类头拆分重组
        s_feats = s_preds[1] if isinstance(s_preds, tuple) else s_preds
        t_feats = t_preds[1] if isinstance(t_preds, tuple) else t_preds
        s_pred_distri, s_pred_scores = self.outend(s_feats)
        t_pred_distri, t_pred_scores = self.outend(t_feats)
        dtype = s_pred_scores.dtype
        batch_size = s_pred_scores.shape[0]
        # 特征图尺寸恢复
        imgsz = torch.tensor(s_feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(s_feats, self.stride, 0.5)
        # GT信息重新整合 坐标信息转换  xywh -> xyxy
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])   # 切片操作，提取两次imgsz
        gt_labels, gt_bboxes = targets.split((1, 4), 2)     # cls, xyxy
        # 每个元素表示相应坐标框的坐标和，然后对上述求和结果主元素比较，将大于0的设置为True，小于等于0的设置为False，标识哪些边界框存在
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        s_pred_bboxes = self.bbox_decode(anchor_points, s_pred_distri)  # xyxy, (b, h*w, 4)
        t_pred_bboxes = self.bbox_decode(anchor_points, t_pred_distri)  # xyxy, (b, h*w, 4)
        # 注意从梯度计算的操作中分离出来
        _, s_target_bboxes, s_target_scores, s_fg_mask, _ = self.assigner(s_pred_scores.detach().sigmoid(), (
                s_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor,
                                                                          gt_labels, gt_bboxes, mask_gt)

        _, t_target_bboxes, t_target_scores, t_fg_mask, _ = self.assigner(t_pred_scores.detach().sigmoid(), (
                t_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor,
                                                                          gt_labels, gt_bboxes, mask_gt)
        t_fg_mask_sum = max(t_fg_mask.sum(), 1)
        if self.cls_loss is not None:   # 只提取正样本的预测分数 求平均
            if self.distiller == 'dkd' or self.distiller == 'nkd' or self.distiller == 'wsld':
                loss[1] = self.cls_loss(s_pred_scores[t_fg_mask], t_pred_scores[t_fg_mask], t_target_scores[t_fg_mask]) / t_fg_mask_sum
            else:
                loss[1] = self.cls_loss(s_pred_scores[t_fg_mask], t_pred_scores[t_fg_mask]) / t_fg_mask_sum

        weight = t_target_scores.sum(-1)[t_fg_mask].unsqueeze(-1)
        if s_fg_mask.sum():
            if self.box_loss is not None:
                if self.distiller == 'bckd':
                    iou = bbox_iou(s_pred_bboxes[t_fg_mask], t_pred_bboxes[t_fg_mask], xywh=False, GIoU=True)
                    if type(iou) is tuple:
                        if len(iou) == 2:
                            loss[0] = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / t_fg_mask_sum
                        else:
                            loss[0] = (iou[0] * iou[1] * weight).sum() / t_fg_mask_sum
                    else:
                        loss[0] = ((1.0 - iou) * weight).sum() / t_fg_mask_sum
                else:
                    loss[0] = self.box_loss(s_pred_bboxes[t_fg_mask], t_pred_bboxes[t_fg_mask])
        loss[0] *= self.box_loss_weight     # box gain
        loss[1] *= self.cls_loss_weight     # cls gain
        return self.loss_weight * loss.sum() * batch_size

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)

    def outend(self, feats):
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)      # 拆分为 64 和 10
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()      # 维度重新排列并确保tensor在内存中的存储是连续的
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        return pred_distri, pred_scores

