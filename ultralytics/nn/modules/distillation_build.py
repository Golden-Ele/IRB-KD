"""
基于特征的蒸馏框架
"""
from __future__ import print_function
import torch.nn.functional as F
import torch
from mmengine.model import constant_init, kaiming_init
from torch import nn
from .utils import weighted_loss

__all__ = ['CWDLoss', 'MGDLoss', 'FSPLoss', 'FGDLoss', 'PKDLoss', 'AWDLoss', 'DMKDLoss']

eps = 1e-7


# DMKDLoss
class DMKDLoss(nn.Module):

    def __init__(self, channels_s, channels_t, alpha_dmkd=0.5, beta_dmkd=0.5, gamma_dmkd=0.001,
                 tao_s=1.00, tao_c=1.00, temp=0.5, temp1=20, area_threshold=500, a=0.00001, b=0.1, c=0.00001):
        super(DMKDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_dmkd = alpha_dmkd
        self.beta_dmkd = beta_dmkd
        self.tao_s = tao_s
        self.tao_c = tao_c
        self.temp = temp
        self.temp1 = temp1  # gt_bbox 面积反比 temperature
        self.gamma_dmkd = gamma_dmkd
        self.area_threshold = area_threshold  # 20x20~32x32
        self.a = a  # mask loss
        self.b = b  # fg_weight_loss
        self.c = c  # bg_loss

        self.conv_mask_s = [nn.Conv2d(channel, 1, kernel_size=1).to(device) for channel_s, channel in
                            zip(channels_s, channels_t)]
        self.conv_mask_t = [nn.Conv2d(channel, 1, kernel_size=1).to(device) for channel_s, channel in
                            zip(channels_s, channels_t)]

        # Spatial Procreate Module
        self.generation_spatial = [
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),     # 3 x 3
                nn.GELU(),                                                 # Swish  激活函数测试
                nn.BatchNorm2d(channel),                                   # BN
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),     # 3 x 3
                nn.GELU(),                                                 # Swish
                nn.BatchNorm2d(channel),                                   # BN
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel_s, channel in
            zip(channels_s, channels_t)                                    # 3 x 3
        ]
        # Channel Procreate Module
        self.generation_channel = [
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1),                # 1 x 1
                nn.GELU(),                                                 # Swish
                nn.LayerNorm([channel, 1, 1]),                             # LN
                nn.Conv2d(channel, channel, kernel_size=1)).to(device) for channel_s, channel in
            zip(channels_s, channels_t)                                    # 1 x 1
        ]

    def forward(self, y_s, y_t, gt_bboxes, layer=None):
        """
        Args:
            y_s (list): The student model prediction with shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)

        loss_area = []      # 存放 area_loss

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if layer == "outlayer":
                idx = -1     # 序列中最后一个元素
            area_loss = self.get_feature_loss(s, t, gt_bboxes, idx)
            loss_area.append(area_loss)
        loss1 = sum(loss_area) * self.gamma_dmkd
        return loss1

    def get_Tmodel_spatial_attention_map(self, preds_T, temp):     # 教师模型的空间注意力图  N*W*H
        N, C, H, W = preds_T.shape
        value = torch.abs(preds_T)
        feat_map = value.mean(axis=1, keepdim=True)      # keepdim = True 表示保持了维度的维数，大小变为1
        teacher_S_attention = (H * W * F.softmax((feat_map / temp).view(N, -1), dim=1)).view(N, H, W)
        return teacher_S_attention.unsqueeze(1)

    def get_Tmodel_channel_attention_map(self, preds_T, temp):      # 教师模型的通道注意力图  N*C
        N, C, H, W = preds_T.shape
        value1 = torch.abs(preds_T)
        channel_map = value1.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
        teacher_C_attention = C * F.softmax(channel_map / temp, dim=1)
        return teacher_C_attention.unsqueeze(2).unsqueeze(3)

    def spatial_pool(self, x, in_type, idx):     # Context Modeling
        N, C, W, H = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(N, C, H * W)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s[idx](x)  # 将通道数置为1
        else:
            context_mask = self.conv_mask_t[idx](x)
        # [N, 1, H * W]
        context_mask = context_mask.view(N, 1, H * W)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)  # dim=2 计算概率分布
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(N, C, 1, 1)

        return context

    def get_feature_loss(self, preds_S, preds_T, gt_bboxes, idx):
        """
        calculate area-weight loss
        """
        loss_mse_sum = nn.MSELoss(reduction='sum')
        loss_mse_none = nn.MSELoss(reduction='none')

        N, C, H, W = preds_T.shape
        device = preds_S.device

        # 定义前景 mask (N, H, W)
        Mask_weight_fg = torch.zeros([N, H, W], dtype=torch.float32, device='cuda')
        # 定义背景 mask (N, H, W)
        Mask_bg = torch.ones([N, H, W], dtype=torch.float32, device='cuda')
        # 定义小目标 mask (N, H, W)
        Mask_tiny_obj = torch.ones([N, H, W], dtype=torch.float32, device='cuda')

        wmin, wmax, hmin, hmax = [], [], [], []  # 存放每个gt_bbox的坐标
        mask_value = []  # 存放每个pixel的权重值
        area_ori = []    # 存放映射到原图的面积

        # 遍历每个batch的每一张图片 进行坐标转换 获取gt_area以及gt_weight
        for i in range(N):
            try:
                if int(N) == 1:
                    batch_bbox = gt_bboxes[1:]
                else:
                    batch_bbox = torch.stack([bbox[1:] for bbox in gt_bboxes if bbox[0] == float(i)])
            except RuntimeError:
                break
            new_boxxes = torch.ones_like(batch_bbox)  # 用于存放转换以后的coordinate
            # 将gt_bboxes映射到相应特征层分辨率
            new_boxxes[:, 0] = batch_bbox[:, 0] * W   # batch_bbox：每个batch中的每张图片的gt_bbox coordinate集合
            new_boxxes[:, 2] = batch_bbox[:, 2] * W
            new_boxxes[:, 1] = batch_bbox[:, 1] * H
            new_boxxes[:, 3] = batch_bbox[:, 3] * H
            # (x_center,y_center,w,h) -> (xyxy)
            wmin.append(torch.floor(new_boxxes[:, 0] - new_boxxes[:, 2] / 2).int())
            wmax.append(torch.ceil(new_boxxes[:, 0] + new_boxxes[:, 2] / 2).int())
            hmin.append(torch.floor(new_boxxes[:, 1] - new_boxxes[:, 3] / 2).int())
            hmax.append(torch.ceil(new_boxxes[:, 1] + new_boxxes[:, 3]).int())
            # 每个gt_bbox在当前feature map下的area
            area = (torch.mul((hmax[i] + 1 - hmin[i]), (wmax[i] + 1 - wmin[i]))).type(torch.float32)

            area_weight = F.softmax((1 - area) / self.temp1, dim=0)  # area-weighted
            area_ori.append((area * 409600) / (H * W))  # 面积转换
            mask_value.append((area_weight / area))   # gt_bbox的每个pixel处的权重

            # Mask_weight_fg的shape(N, C, H, W) 获取前景的面积加权图
            for j in range(len(batch_bbox)):
                Mask_weight_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = mask_value[i][j]
                Mask_bg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = 0
                if area_ori[i][j] < self.area_threshold:  # 进行每个gt_area的阈值判断 只mask小目标区域
                    Mask_tiny_obj[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = 0

        # 获取 spatial-attention mask + tiny object mask
        mat_s = self.get_Tmodel_spatial_attention_map(preds_T, temp=0.5)  # (N, 1, H, W)

        mat_s = torch.where(mat_s >= self.tao_s, 0, 1).to(device).type(torch.float16)  # spatial_mask (N, 1, H, W)

        Mask_tiny_obj = Mask_tiny_obj.unsqueeze(1).to(device).type(torch.float16)  # (N, H, W) -> (N, 1, H, W)
        mat_total = torch.mul(mat_s, Mask_tiny_obj)  # 0为需要mask的区域
        masked_fea1 = torch.mul(preds_S, mat_total)  # mask操作
        new_fea1 = self.generation_spatial[idx](masked_fea1)  # masked后的feature尝试生成teacher的feature maps

        # 获取 channel-attention mask
        mat_c = self.get_Tmodel_channel_attention_map(preds_T, temp=0.5)  # (N, C, 1, 1)
        mat_c = torch.where(mat_c >= self.tao_c, 0, 1).to(device)  # channel_mask
        masked_fea2 = torch.mul(preds_S, mat_c)  # mask操作
        masked_fea2_2 = self.spatial_pool(masked_fea2, 0, idx)  # Context Modeling -> LayerNorm
        new_fea2 = self.generation_channel[idx](masked_fea2_2)  # masked后的feature尝试生成teacher的feature maps

        f_rec = self.alpha_dmkd * new_fea1 + self.beta_dmkd * new_fea2

        mask_loss = loss_mse_sum(preds_T, f_rec) / N      # mask loss

        Mask_weight_fg = Mask_weight_fg.unsqueeze(1)
        Mask_bg = Mask_bg.unsqueeze(1)
        # 计算fg_weighted loss
        ori_loss = loss_mse_none(preds_T, preds_S)        # ori loss
        fg_weight_loss = torch.mul(ori_loss, Mask_weight_fg)
        fg_weight_loss = torch.sum(fg_weight_loss) / N       # fg_weight_loss
        bg_loss = torch.mul(ori_loss, Mask_bg)
        bg_loss = torch.sum(bg_loss) / N                # bg_loss
        # (1-a-b)的超参数设置
        loss_feature = self.a * mask_loss + self.b * fg_weight_loss + self.c * bg_loss  # total loss
        # loss_feature = self.a * mask_loss                           # only mask loss
        # loss_feature = self.b * fg_weight_loss + self.c * bg_loss   # only decoupled loss

        return loss_feature


# AWD
class AWDLoss(nn.Module):

    def __init__(self, channels_s, channels_t, alpha_awd=0.00001, lambda_awd=1.00, temp=0.5):
        super(AWDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_awd = alpha_awd
        self.lambda_awd = lambda_awd
        self.temp = temp

        # 原论文中的生成块 用于生成教师模型的feature maps; 原论文经实验，两个3x3卷积效果最好
        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel_s, channel in
            zip(channels_s, channels_t)
        ]

    def forward(self, y_s, y_t, layer=None):
        """
        Args:
            y_s (list): The student model prediction with shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)  # assert(断言)，用于判断一个表达式，在表达式条件为false的时候触发异常
        losses = []  # 用于存放每个阶段的loss
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # print(s.shape)
            # print(t.shape)
            # assert s.shape == t.shape
            if layer == "outlayer":
                idx = -1  # 序列中最后一个元素
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_awd)
        loss = sum(losses)
        return loss

    def get_Tmodel_attention(self, preds_T, temp):
        N, C, H, W = preds_T.shape

        value = torch.abs(preds_T)
        feat_map = value.mean(axis=1, keepdim=True)
        teacher_attention = (H * W * F.softmax((feat_map / temp).view(N, -1), dim=1)).view(N, H, W)
        return teacher_attention.unsqueeze(1)

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')  # 表示将所有损失值相加得到一个标量值，用于衡量整体的损失 L2
        N, C, H, W = preds_T.shape

        # 这段代码创建了一个随机的(N,1,H,W)形状的张量mat，然后根据条件对其元素进行替换，使得满足条件的元素变为0，不满足条件的元素变为1
        # 这种操作在一些情况下可以用来生成随机的二值化掩模或掩码
        device = preds_S.device
        mat = self.get_Tmodel_attention(preds_T, temp=0.5)
        # 如果condition中的元素为True，则选择x中的对应元素；否则选择y中的对应元素。这将创建一个新的tensor，形状与condition相同
        mat = torch.where(mat > self.lambda_awd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)  # mask
        new_fea = self.generation[idx](masked_fea)  # masked后的feature尝试生成teacher的feature maps
        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss


#   FGD
class FGDLoss(nn.Module):
    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`

    Args:
        student_channels(list): Number of channels in the student's feature map.
        teacher_channels(list): Number of channels in the teacher's feature map.
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """

    def __init__(self, student_channels, teacher_channels, temp=0.5, alpha_fgd=0.001, beta_fgd=0.0005, gamma_fgd=0.001,
                 lambda_fgd=0.000005):
        super(FGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        self.conv_mask_s = [nn.Conv2d(teacher_channel, 1, kernel_size=1).to(device) for teacher_channel in
                            teacher_channels]
        self.conv_mask_t = [nn.Conv2d(teacher_channel, 1, kernel_size=1).to(device) for teacher_channel in
                            teacher_channels]

        self.channel_add_conv_s = [nn.Sequential(
            nn.Conv2d(teacher_channel, teacher_channel // 2, kernel_size=1),
            nn.LayerNorm([teacher_channel // 2, 1, 1]),
            nn.ReLU(inplace=True),   # yapf: disable
            nn.Conv2d(teacher_channel // 2, teacher_channel, kernel_size=1)).to(device) for teacher_channel in
                                   teacher_channels]
        self.channel_add_conv_t = [nn.Sequential(
            nn.Conv2d(teacher_channel, teacher_channel // 2, kernel_size=1),
            nn.LayerNorm([teacher_channel // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channel // 2, teacher_channel, kernel_size=1)).to(device) for teacher_channel in
                                   teacher_channels]

        self.reset_parameters()

    def forward(self, preds_S, preds_T, gt_bboxes, img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            一个 batch 的 Ground truth bboxes 的 list, 每个图片 gt labels 的形状为 (num_gts,)
            img_metas (list[dict]): Meta information of each image, e.g., image size, scaling factor, etc.
            一个 batch 的 image 的信息的 list, 如: 大小, 缩放等.
        """
        assert len(preds_S) == len(preds_T), 'the output dim of teacher and student differ'
        losses = []
        for idx, (s, t) in enumerate(zip(preds_S, preds_T)):
            losses.append(self.get_dis_loss(s, t, gt_bboxes, img_metas, idx))
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, gt_bboxes, img_metas, idx):
        N, C, H, W = preds_S.shape
        # 获取相应的注意力机制图
        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)
        # 生成相应的注意力机制掩码  Bs*W*H 空间
        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)
        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(N):
            try:
                if int(N) == 1:
                    batch_bbox = gt_bboxes[1:]
                else:
                    batch_bbox = torch.stack([bbox[1:] for bbox in gt_bboxes if bbox[0] == float(i)])
            except RuntimeError:
                break
            # batch_bbox = torch.stack([bbox[1:] for bbox in gt_bboxes if bbox[0] == float(i)])
            new_boxxes = torch.ones_like(batch_bbox)

            new_boxxes[:, 0] = batch_bbox[:, 0] * W
            new_boxxes[:, 2] = batch_bbox[:, 2] * W
            new_boxxes[:, 1] = batch_bbox[:, 1] * H
            new_boxxes[:, 3] = batch_bbox[:, 3] * H
            wmin.append(torch.floor(new_boxxes[:, 0] - new_boxxes[:, 2] / 2).int())
            wmax.append(torch.ceil(new_boxxes[:, 0] + new_boxxes[:, 2] / 2).int())
            hmin.append(torch.floor(new_boxxes[:, 1] - new_boxxes[:, 3] / 2).int())
            hmax.append(torch.ceil(new_boxxes[:, 1] + new_boxxes[:, 3]).int())

            # 表示 面积的倒数
            area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
                    wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))

            for j in range(len(batch_bbox)):  # 返回第0维长度
                Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
                    torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i] > 0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg,
                                             C_attention_s, C_attention_t, S_attention_s, S_attention_t)

        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T, idx)

        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss

        return loss

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W = preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
        C_attention = C * F.softmax(channel_map / temp, dim=1)

        return S_attention, C_attention

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s - C_t))) / len(C_s) + torch.sum(torch.abs((S_s - S_t))) / len(S_s)

        return mask_loss

    def spatial_pool(self, x, in_type, idx):        # GcBlock
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s[idx](x)
        else:
            context_mask = self.conv_mask_t[idx](x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def get_rela_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0, idx)
        context_t = self.spatial_pool(preds_T, 1, idx)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s[idx](context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t[idx](context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t) / len(out_s)

        return rela_loss

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        for i in range(len(self.conv_mask_s)):
            self.conv_mask_s[i].inited = True
        for i in range(len(self.conv_mask_t)):
            self.conv_mask_t[i].inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)


#   CWD
class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape

            # 对teacher网络的输出softmax化
            # softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]   channel-wise
            softmax_pred_T = F.softmax(t.view(N * W * H, C) / self.tau, dim=1)  # [N*C, H*W]   spatial-wise

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(N * W * H, C) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(N * W * H, C) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (N * W * H))
        loss = sum(losses)

        return loss


#   MGD
class MGDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel in
            channels_t
        ]

    def forward(self, y_s, y_t, layer=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # print(s.shape)
            # print(t.shape)
            # assert s.shape == t.shape
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss


#   FSP
class FSPLoss(nn.Module):
    """A Gift from Knowledge Distillation:
    Fast Optimization, Network Minimization and Transfer Learning"""

    def __init__(self, s_shapes, t_shapes):
        super(FSPLoss, self).__init__()

    def forward(self, g_s, g_t):
        assert len(g_s) == len(g_t)
        s_fsp = self.compute_fsp(g_s)
        t_fsp = self.compute_fsp(g_t)
        loss_group = [self.compute_loss(s, t) for s, t in zip(s_fsp, t_fsp)]
        loss = sum(loss_group)
        return loss

    @staticmethod
    def compute_loss(s, t):
        return (s - t).pow(2).mean()

    @staticmethod
    def compute_fsp(g):
        fsp_list = []
        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass
            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)

            fsp = (bot * top).mean(-1)
            fsp_list.append(fsp)
        return fsp_list


#   PKD
def norm(feat: torch.Tensor) -> torch.Tensor:
    """
    Args:
        feat (torch.Tensor): The original feature map with shape
            (N, C, H, W).
    """
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)  # (N, C, H, W) -> (C, N, H, W) -> (C, N*H*W)
    # 计算每个channel的均值和方差
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)  # 使每个通道的值分布接近标准正态分布
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)  # 还原feature map


@weighted_loss
def pkd_loss(pred, target):
    pred = norm(pred)
    target = norm(target)
    return F.mse_loss(pred, target, reduction='none') / 2


class PKDLoss(nn.Module):

    def __init__(self, channels_s, channels_t, reduction='mean', loss_weight=1.0):
        super(PKDLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, y_s, y_t, weight=None, avg_factor=None, reduction_override=None) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            loss = pkd_loss(s, t, weight, reduction=reduction, avg_factor=avg_factor)
            losses.append(loss)
        return self.loss_weight * sum(losses)
