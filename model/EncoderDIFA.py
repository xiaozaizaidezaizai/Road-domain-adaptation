import torch
import torch.nn as nn
from torchvision import models

from DoMain_12.model.DiFa import DIFA


class ResNet34EncoderWithDIFA(nn.Module):
    """
    集成了 DIFA 模块的 ResNet34 编码器
    用于【域不变编码器】分支
    """

    def __init__(self, pretrained=True):
        super(ResNet34EncoderWithDIFA, self).__init__()
        # 加载基础 ResNet34
        original_model = models.resnet34(pretrained=pretrained)

        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1  # 64 channels
        self.layer2 = original_model.layer2  # 128 channels
        self.layer3 = original_model.layer3  # 256 channels
        self.layer4 = original_model.layer4  # 512 channels

        # 初始化 DIFA 模块
        # 插入到 layer2, layer3, layer4 之后
        self.difa2 = DIFA(128)
        self.difa3 = DIFA(256)
        self.difa4 = DIFA(512)

    def forward(self, x, alpha=1.0):
        """
        Args:
            x: Input image
            alpha: GRL 的参数 (随着训练进行动态调整)
        Returns:
            features: [f1, f2, f3, f4] (用于解码器)
            domain_preds: [d2, d3, d4] (用于计算对抗损失)
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1 (通常不加对抗，因为特征太浅)
        f1 = self.layer1(x)

        # Layer 2 + DIFA
        f2_orig = self.layer2(f1)
        # 将 f2 传入 DIFA，得到增强后的特征 f2_enh 和 域预测 d2
        f2_enh, d2 = self.difa2(f2_orig, alpha)

        # Layer 3 + DIFA
        # 注意：这里我们将增强后的 f2_enh 传入 layer3，实现了"反馈到主干"
        f3_orig = self.layer3(f2_enh)
        f3_enh, d3 = self.difa3(f3_orig, alpha)

        # Layer 4 + DIFA
        f4_orig = self.layer4(f3_enh)
        f4_enh, d4 = self.difa4(f4_orig, alpha)

        # 返回增强后的特征列表给解码器使用
        features = [f1, f2_enh, f3_enh, f4_enh]

        # 返回域预测列表给 Loss 计算使用
        domain_preds = [d2, d3, d4]

        return features, domain_preds


if __name__ == '__main__':
    # 测试代码
    model = ResNet34EncoderWithDIFA(pretrained=False)
    dummy_input = torch.randn(2, 3, 512, 512)

    feats, preds = model(dummy_input, alpha=0.5)

    print("Features shapes:")
    for f in feats:
        print(f.shape)

    print("\nDomain Predictions shapes:")
    for p in preds:
        print(p.shape)

    # 预期输出:
    # Features: [B, 64, 128, 128], [B, 128, 64, 64], [B, 256, 32, 32], [B, 512, 16, 16]
    # Predictions: [B, 1, H/4, W/4]... 取决于判别器最后的分辨率