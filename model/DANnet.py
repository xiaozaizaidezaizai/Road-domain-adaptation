import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# from EncoderDIFA import ResNet34EncoderWithDIFA

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


class ResNet34Encoder(nn.Module):
    """
    基于 ResNet34 的编码器
    返回多尺度特征列表 [feat1, feat2, feat3, feat4] 用于多级对齐
    """

    def __init__(self, pretrained=True):
        super(ResNet34Encoder, self).__init__()
        # 加载预训练的 resnet34
        original_model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)

        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)  # 64 channels, 1/4 size
        f2 = self.layer2(f1)  # 128 channels, 1/8 size
        f3 = self.layer3(f2)  # 256 channels, 1/16 size
        f4 = self.layer4(f3)  # 512 channels, 1/32 size

        # 返回所有层的特征，方便后续的多级对齐(DIFA)和解码
        return [f1, f2, f3, f4]


class DecoderBlock(nn.Module):
    """ 基本的解码器块：上采样 + 卷积 + BN + ReLU """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DAN_Net_Decoder(nn.Module):
    """
    解码器
    接收编码器的特征，使用双重注意力模块，并逐步上采样恢复分辨率
    """

    def __init__(self, num_classes=1):
        super(DAN_Net_Decoder, self).__init__()

        # ResNet34 的通道数: layer4=512, layer3=256, layer2=128, layer1=64

        # 双重注意力模块放置在最深层特征 (f4) 之后
        self.dam = DualAttentionModule(512)

        # 解码层 (类似 U-Net 结构，自底向上融合)
        # F4 (512) -> Upsample -> Concat F3 (256) -> Decode
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DecoderBlock(512 + 256, 256)  # 512来自F4上采样, 256来自F3

        # F3_dec (256) -> Upsample -> Concat F2 (128) -> Decode
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DecoderBlock(256 + 128, 128)

        # F2_dec (128) -> Upsample -> Concat F1 (64) -> Decode
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DecoderBlock(128 + 64, 64)

        # 最后的分类层
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 恢复到原图大小 (因为layer1是1/4)

    def forward(self, features):
        # features 是一个列表 [f1, f2, f3, f4]
        f1, f2, f3, f4 = features

        # 1. 在最深层应用双重注意力
        x = self.dam(f4)

        # 2. 逐级上采样和融合 (Skip Connections)
        x = self.up4(x)
        x = torch.cat([x, f3], dim=1)  # 融合 F3
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, f2], dim=1)  # 融合 F2
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)  # 融合 F1
        x = self.dec2(x)

        # 3. 最终预测
        out = self.final_conv(x)
        out = self.final_up(out)

        return out


class DAN_Net(nn.Module):
    """
    DAN-Net 总体结构
    包含：
    1. 域不变编码器 (Invariant Encoder)
    2. 域特定编码器 (Specific Encoder)
    3. 解码器 (Decoder)
    """

    def __init__(self, num_classes=1, pretrained=True):
        super(DAN_Net, self).__init__()

        # 定义两个独立的编码器
        self.invariant_encoder = ResNet34EncoderWithDIFA(pretrained=pretrained)
        self.specific_encoder = ResNet34Encoder(pretrained=pretrained)

        # 定义解码器
        self.decoder = DAN_Net_Decoder(num_classes=num_classes)

        # 注意：这里只定义了基础结构。
        # 对抗对齐模块(DIFA)、SFDP 和 目标增强模块(TEM) 通常在训练循环或Forward逻辑中
        # 通过 Loss 函数和特征交互来实现，或者作为独立的子模块挂载。

    def forward(self, x, domain='source'):
        """
        Forward 函数根据训练阶段可能有不同的逻辑。
        这里提供一个基础的流程。
        """
        # 1. 获取域不变特征 (所有图像都通过这个分支)
        inv_features,domain_preds = self.invariant_encoder(x)

        # 2. 获取域特定特征 (通常目标域图像通过这个分支)
        spec_features = self.specific_encoder(x)

        # 3. 解码 (主要使用域不变特征进行预测)
        # 在论文提到的"对比学习"阶段，可能需要将 inv_features 和 spec_features 融合后再解码
        # 这里默认返回基于域不变特征的预测结果

        pred_inv = self.decoder(inv_features)

        # 如果需要返回特征用于计算对抗损失或对比损失，可以在这里返回
        return pred_inv, inv_features, spec_features


if __name__ == '__main__':
    # 测试代码
    model = DAN_Net()
    dummy_input = torch.randn(2, 3, 512, 512)  # Batch size 2, RGB, 512x512
    output, inv_feat, spec_feat = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)  # 应该是 [2, 1, 512, 512]
    print("Invariant Features (Layer 4) shape:", inv_feat[-1].shape)
    print("Specific Features (Layer 4) shape:", spec_feat[-1].shape)