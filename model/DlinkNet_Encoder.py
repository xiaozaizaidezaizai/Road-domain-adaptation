import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from DoMain_12.model.DiFa import DIFA,DIFA1,DIFADecoderBlock
# 假设你已经定义了 DIFA 类，如果没有，请确保导入它
# from model.DIFA import DIFA

class Dblock(nn.Module):
    """
    D-LinkNet 的核心：中心膨胀卷积模块
    通过串联不同膨胀率的卷积层来扩大感受野，同时保持分辨率
    """

    def __init__(self, channel):
        super(Dblock, self).__init__()
        # 膨胀率分别为 1, 2, 4, 8
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

        # 所有的卷积层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # 串联结构 (Cascade) + 残差连接
        d1 = self.relu(self.dilate1(x))
        d2 = self.relu(self.dilate2(d1))
        d3 = self.relu(self.dilate3(d2))
        d4 = self.relu(self.dilate4(d3))

        # 将所有尺度的特征相加，这使得网络能同时感知局部和全局信息
        out = x + d1 + d2 + d3 + d4
        return out


class DLinkNetEncoderWithDIFA(nn.Module):
    """
    集成了 DIFA 模块和 Dblock 的 ResNet34 编码器
    结构: ResNet34 Layers + DIFA Interleaving + Dblock Center
    """

    def __init__(self, pretrained=True):
        super(DLinkNetEncoderWithDIFA, self).__init__()

        # 1. 加载基础 ResNet34
        resnet = models.resnet34(pretrained=pretrained)

        # 初始层 (Stem)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 编码层
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

        # 2. DIFA 模块 (用于域适应对抗)
        # 插入到 layer2, layer3, layer4 之后
        # 注意：你需要确保外部已经定义了 DIFA 类
        self.difa2 = DIFA(128)
        self.difa3 = DIFA(256)# DIFA1是没有判别器的
        self.difa4 = DIFA(512)

        # 3. D-LinkNet 中心膨胀模块
        # 放在 Layer4 之后，处理最高维特征
        self.center_dblock = Dblock(512)

    def forward(self, x, alpha=1.0):
        """
        Args:
            x: 输入图像
            alpha: GRL 参数
        Returns:
            features: [f1, f2, f3, center_feat] (用于解码器)
            domain_preds: [d2, d3, d4] (用于对抗损失)
        """
        # --- ResNet Stem ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # --- Layer 1 ---
        # 尺寸: H/4, W/4, Channel: 64
        f1 = self.layer1(x)

        # --- Layer 2 + DIFA ---
        # 原始 f2
        f2_orig = self.layer2(f1)
        # DIFA 增强 & 域预测
        f2_enh,d2 = self.difa2(f2_orig, alpha)

        # --- Layer 3 + DIFA ---
        # 关键点：将 f2_enh 传入 layer3 (反馈机制)
        f3_orig = self.layer3(f2_enh)
        f3_enh,d3 = self.difa3(f3_orig, alpha)

        # --- Layer 4 + DIFA ---
        f4_orig = self.layer4(f3_enh)
        f4_enh, d4 = self.difa4(f4_orig, alpha)

        # --- D-LinkNet Center Block ---
        # 对 layer4 的输出进行膨胀卷积增强
        # center_feat = self.center_dblock(f4_enh)

        # 返回特征列表 (Skip Connections)
        # 注意：这里我们返回的是 center_feat 替代原来的 f4
        features = [f1, f2_enh, f3_enh, f4_enh]

        # 返回域预测用于计算 Loss
        domain_preds = [d2,d3,d4]

        return features, domain_preds

        # # 单层判别器
        # f2_enh = self.difa2(f2_orig, alpha)
        #
        # # --- Layer 3 + DIFA ---
        # # 关键点：将 f2_enh 传入 layer3 (反馈机制)
        # f3_orig = self.layer3(f2_enh)
        # f3_enh = self.difa3(f3_orig, alpha)
        #
        # # --- Layer 4 + DIFA ---
        # f4_orig = self.layer4(f3_enh)
        # f4_enh, d4 = self.difa4(f4_orig, alpha)
        #
        # # --- D-LinkNet Center Block ---
        # # 对 layer4 的输出进行膨胀卷积增强
        # center_feat = self.center_dblock(f4_enh)
        #
        # # 返回特征列表 (Skip Connections)
        # # 注意：这里我们返回的是 center_feat 替代原来的 f4
        # features = [f1, f2_enh, f3_enh, center_feat]
        #
        # # 返回域预测用于计算 Loss
        # domain_preds = [d4]
        #
        # return features, domain_preds

# -------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    D-LinkNet 解码器中的基础卷积块
    """

    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class PositionAttentionModule(nn.Module):
    """ 位置注意力模块 (捕获空间依赖) """

    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()
        self.chanel_in = in_channels

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class ChannelAttentionModule(nn.Module):
    """ 通道注意力模块 (捕获通道依赖) """

    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class DualAttentionModule(nn.Module):
    """
    双重注意力模块 (Dual Attention Module)
    结合了位置注意力 (Position Attention) 和通道注意力 (Channel Attention)
    用于解码器部分，增强全局和局部特征的捕获能力。
    """

    def __init__(self, in_channels):
        super(DualAttentionModule, self).__init__()
        self.pam = PositionAttentionModule(in_channels)
        self.cam = ChannelAttentionModule(in_channels)
        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        feat_pam = self.pam(x)
        feat_cam = self.cam(x)
        # 将两种注意力的特征融合 (这里采用相加)
        feat_sum = feat_pam + feat_cam
        out = self.conv_out(feat_sum)
        out = self.bn(out)
        return self.relu(out)

class DLinkNetDecoder(nn.Module):
    """
    D-LinkNet 的解码器部分
    接收编码器的多尺度特征，通过跳跃连接恢复分辨率
    """

    def __init__(self, num_classes=1):
        super(DLinkNetDecoder, self).__init__()

        # 定义解码器块
        # ResNet34 filters: [64, 128, 256, 512]
        self.dam = DualAttentionModule(512)
        # Stage 4: Input 512 (Center) + Skip 256 (Layer3) -> Output 256
        self.decoder4 = DecoderBlock(512, 256)

        # Stage 3: Input 256 (Dec4) + Skip 128 (Layer2) -> Output 128
        self.decoder3 = DecoderBlock(256 + 256, 128)  # Cat后通道增加

        # Stage 2: Input 128 (Dec3) + Skip 64 (Layer1) -> Output 64
        self.decoder2 = DecoderBlock(128 + 128, 64)

        # Stage 1: Input 64 (Dec2) -> Output 64 (恢复到 H/2, W/2)
        self.decoder1 = DecoderBlock(64 + 64, 64)

        # Final: 恢复到 H, W 并输出分类
        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, features):
        """
        Args:
            features: 编码器返回的列表 [f1, f2, f3, center_feat]
                      f1: 64, H/4
                      f2: 128, H/8
                      f3: 256, H/16
                      center: 512, H/32
        """
        f1, f2, f3, center = features

        # --- Decoder Path ---

        center = self.dam(center)

        # 1. Center (512) -> Upsample -> (256)
        d4 = self.decoder4(center)
        # d4: 256, H/16

        # 2. Concat(d4, f3) -> (256+256) -> Decoder3 -> (128)
        d3 = self.decoder3(torch.cat([d4, f3], dim=1))
        # d3: 128, H/8

        # 3. Concat(d3, f2) -> (128+128) -> Decoder2 -> (64)
        d2 = self.decoder2(torch.cat([d3, f2], dim=1))
        # d2: 64, H/4

        # 4. Concat(d2, f1) -> (64+64) -> Decoder1 -> (64)
        d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        # d1: 64, H/2

        # 5. Final Upsample -> H, W
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


class DLinkNetDecoderWithDIFA(nn.Module):
    """
    基于 DIFA 的 D-LinkNet 解码器
    特点：
    1. 保留了 Center 的 DualAttentionModule (DAM)
    2. 将普通解码块替换为 DIFADecoderBlock
    """

    def __init__(self, num_classes=1):
        super(DLinkNetDecoderWithDIFA, self).__init__()

        # --- 核心保留：通道空间注意力模块 ---
        # 放在 Bottleneck 处，处理最抽象的特征
        self.dam = DualAttentionModule(512)

        # --- 替换为 DIFA 解码块 ---

        # Stage 4: Center (512) -> Output (256)
        # 输入只有 center feature
        self.decoder4 = DIFADecoderBlock(512, 256)

        # Stage 3: Input (256 from d4 + 256 from f3) = 512 -> Output (128)
        self.decoder3 = DIFADecoderBlock(256 + 256, 128)

        # Stage 2: Input (128 from d3 + 128 from f2) = 256 -> Output (64)
        self.decoder2 = DIFADecoderBlock(128 + 128, 64)

        # Stage 1: Input (64 from d2 + 64 from f1) = 128 -> Output (64)
        self.decoder1 = DIFADecoderBlock(64 + 64, 64)

        # --- Final Head (保持不变) ---
        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, center_feat]
        """
        f1, f2, f3, center = features

        # 1. 核心注意力增强 (保留)
        center = self.dam(center)

        # 2. 解码路径 (使用 DIFA 增强)

        # Block 4: 512 -> 256
        d4 = self.decoder4(center)
        # d4 shape: [B, 256, H/16, W/16]

        # Block 3: Cat(d4, f3) -> 512 -> 128
        d3 = self.decoder3(torch.cat([d4, f3], dim=1))
        # d3 shape: [B, 128, H/8, W/8]

        # Block 2: Cat(d3, f2) -> 256 -> 64
        d2 = self.decoder2(torch.cat([d3, f2], dim=1))
        # d2 shape: [B, 64, H/4, W/4]

        # Block 1: Cat(d2, f1) -> 128 -> 64
        d1 = self.decoder1(torch.cat([d2, f1], dim=1))
        # d1 shape: [B, 64, H/2, W/2]

        # 3. 输出层
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out