import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class GradientReverseLayer(torch.autograd.Function):
    """
    梯度反转层 (Gradient Reversal Layer)
    前向传播：恒等变换
    反向传播：梯度乘以 -alpha
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, alpha=1.0):
    return GradientReverseLayer.apply(x, alpha)


class CBR(nn.Module):
    """ Conv + BatchNorm + ReLU """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SpatialDomainPerception(nn.Module):
    """
    空间域感知 (SDP)
    结构: CBR(3x3) -> CBR(3x3) -> CBR(3x3)
    通道变化: C -> C/2 -> C/2 -> C
    """

    def __init__(self, in_channels):
        super(SpatialDomainPerception, self).__init__()
        mid_channels = in_channels // 2
        self.block = nn.Sequential(
            CBR(in_channels, mid_channels, kernel_size=3, padding=1),
            CBR(mid_channels, mid_channels, kernel_size=3, padding=1),
            CBR(mid_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class FrequencyDomainPerception(nn.Module):
    """
    频域感知 (FDP)
    结构: FFT -> Conv1x1(Learn Mask) -> Multiply -> IFFT
    """

    def __init__(self, in_channels):
        super(FrequencyDomainPerception, self).__init__()
        # FFT 后我们会得到实部和虚部，拼接后通道数为 2 * in_channels
        self.freq_conv = nn.Sequential(
            # 输入通道翻倍因为包含 real 和 imag 部分
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        # 1. FFT 变换 (Real -> Complex)
        # rfft2 输出形状: [B, C, H, W/2 + 1] (复数)
        fft_feat = torch.fft.rfft2(x, norm='ortho')

        # 2. 拼接实部和虚部以进行卷积处理
        # shape: [B, C*2, H, W/2 + 1]
        real = fft_feat.real
        imag = fft_feat.imag
        cat_feat = torch.cat([real, imag], dim=1)

        # 3. 学习频域掩码 (Mask)
        mask = self.freq_conv(cat_feat)

        # 4. 将掩码作用于频域特征
        # 分离掩码为实部和虚部权重
        mask_real, mask_imag = torch.chunk(mask, 2, dim=1)

        # 加权 (这里采用简单的乘法，也可以尝试复数乘法逻辑)
        # 增强后的实部和虚部
        out_real = real * mask_real
        out_imag = imag * mask_imag

        # 重组为复数
        fft_enhanced = torch.complex(out_real, out_imag)

        # 5. IFFT 还原 (Complex -> Real)
        # 指定输出大小 s=(H, W) 以处理奇偶尺寸问题
        out = torch.fft.irfft2(fft_enhanced, s=(H, W), norm='ortho')

        return out


class SFDP(nn.Module):
    """
    空间和频域感知融合模块
    Output = SDP(x) + FDP(x)
    """

    def __init__(self, in_channels):
        super(SFDP, self).__init__()
        self.sdp = SpatialDomainPerception(in_channels)
        self.fdp = FrequencyDomainPerception(in_channels)

    def forward(self, x):
        # 空间特征
        s_feat = self.sdp(x)
        # 频域特征
        f_feat = self.fdp(x)
        # 融合 (相加)
        return s_feat + f_feat


class DomainDiscriminator(nn.Module):
    """
    域判别器 (参考图中结构)
    包含 Concat 操作的密集连接块
    """

    def __init__(self, in_channels):
        super(DomainDiscriminator, self).__init__()

        # 降维/特征变换: Con 1x1 (图中 GRL 前的那个)
        self.pre_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.bn_pre = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)

        dim = in_channels // 2

        # 判别器主体 (模拟图中多层 Concat 结构)
        # Block 1
        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.LeakyReLU(0.2))
        # Block 2 (Input: dim + dim) -> dim
        self.conv2 = nn.Sequential(nn.Conv2d(dim * 2, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.LeakyReLU(0.2))
        # Block 3 (Input: dim + dim) -> dim
        self.conv3 = nn.Sequential(nn.Conv2d(dim * 2, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.LeakyReLU(0.2))

        # Final Classifier
        self.final = nn.Conv2d(dim, 1, kernel_size=1)  # 输出域置信度图

    def forward(self, x, alpha=1.0):
        # 1. 预处理
        x = self.relu(self.bn_pre(self.pre_conv(x)))

        # 2. 梯度反转
        x = grad_reverse(x, alpha)

        # 3. 判别器推理
        f1 = self.conv1(x)

        # Concat 1
        c1 = torch.cat([x, f1], dim=1)
        f2 = self.conv2(c1)

        # Concat 2
        c2 = torch.cat([f1, f2], dim=1)  # 或者 cat([x, f1, f2])? 这里假设级联
        f3 = self.conv3(c2)

        # 4. 输出预测
        out = self.final(f3)
        return out


class DIFA(nn.Module):
    """
    域不变特征对齐模块 (Domain-Invariant Feature Alignment)
    集成 SFDP 和 Discriminator
    """

    def __init__(self, in_channels):
        super(DIFA, self).__init__()
        self.sfdp = SFDP(in_channels)
        self.discriminator = DomainDiscriminator(in_channels)

        # 融合系数 (可选，如果想让原始特征和增强特征加权融合)
        # 这里直接输出增强后的特征

    def forward(self, x, alpha=1.0):
        """
        Returns:
            enhanced_feat: 经过 SFDP 增强后的特征 (用于传给下一层)
            domain_pred: 域判别结果 (用于计算 L_adv)
        """
        # 1. 特征增强 (SFDP)
        # 论文提到: "将解耦后的特征反馈到主干上"
        # 这里的 enhanced_feat 将替换 ResNet 原始输出传入下一层
        feature_enhanced = self.sfdp(x)

        # 也可以选择残差连接: return x + feature_enhanced
        # 根据图示 (+) 在 SFDP 内部已经完成 (SDP+FDP)
        # 假设 SFDP 输出就是我们要的增强特征

        # 2. 域判别
        domain_pred = self.discriminator(feature_enhanced, alpha)

        return feature_enhanced, domain_pred

class DIFA1(nn.Module):
    """
    域不变特征对齐模块 (Domain-Invariant Feature Alignment)
    集成 SFDP 和 Discriminator
    """

    def __init__(self, in_channels):
        super(DIFA1, self).__init__()
        self.sfdp = SFDP(in_channels)
        # self.discriminator = DomainDiscriminator(in_channels)

        # 融合系数 (可选，如果想让原始特征和增强特征加权融合)
        # 这里直接输出增强后的特征

    def forward(self, x, alpha=1.0):
        """
        Returns:
            enhanced_feat: 经过 SFDP 增强后的特征 (用于传给下一层)
            domain_pred: 域判别结果 (用于计算 L_adv)
        """
        # 1. 特征增强 (SFDP)
        # 论文提到: "将解耦后的特征反馈到主干上"
        # 这里的 enhanced_feat 将替换 ResNet 原始输出传入下一层
        feature_enhanced = self.sfdp(x)

        # 也可以选择残差连接: return x + feature_enhanced
        # 根据图示 (+) 在 SFDP 内部已经完成 (SDP+FDP)
        # 假设 SFDP 输出就是我们要的增强特征



        return feature_enhanced


class DIFADecoderBlock(nn.Module):
    """
    集成了 DIFA 增强的解码块
    流程: 降维/融合 -> DIFA增强 -> 上采样
    """

    def __init__(self, in_channels, out_channels):
        super(DIFADecoderBlock, self).__init__()

        # 1. 降维/特征融合 (Conv 1x1)
        # 先把通道数降到目标维度，减少后续 DIFA 的计算量
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 2. DIFA 模块 (核心替换部分)
        # 使用 DIFA1，因为解码器主要用于特征恢复，不需要输出 Domain Prediction
        self.difa = DIFA1(out_channels)

        # 3. 上采样 (转置卷积)
        # 将特征图尺寸放大 2 倍
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Step 1: 调整通道
        x = self.reduce_conv(x)

        # Step 2: 频域+空域 增强 (修复断裂)
        x = self.difa(x)

        # Step 3: 恢复尺寸
        x = self.upsample(x)
        x = self.relu(x)
        return x