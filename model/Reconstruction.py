import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionModule(nn.Module):
    """
    RP (Reconstruction Process) 模块
    功能：将解耦的'域不变特征'和'域特定特征'融合并重建为原始图像，
          以确保特征保留了目标域的完整信息。
    结构参考：Image 1x1conv-128 -> 3x3conv-64 -> 1x1conv-3
    """

    def __init__(self, in_channels=512):
        super(ReconstructionModule, self).__init__()

        # 1. 特征降维与融合处理
        # Input: (B, in_channels, H_feat, W_feat)
        # 对应图中: 1x1conv-128 -> BN/ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 2. 特征提取
        # 对应图中: 3x3conv-64 -> BN/ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 3. 图像重构
        # 对应图中: 1x1conv-3
        # 输出 3 通道对应 RGB 图像
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1, bias=True)

    def forward(self, feat_inv, feat_spec, target_size=None):
        """
        Args:
            feat_inv: 域不变特征 (B, C, H, W)
            feat_spec: 域特定特征 (B, C, H, W)
            target_size: 原始输入图像的大小 (H_img, W_img)，用于上采样
        Returns:
            recon_img: 重建的图像
        """
        # 1. 融合 (Fusion)
        # 图中加号 (+) 表示 Element-wise Add
        fused_feat = feat_inv + feat_spec

        # 2. 网络前向传播
        x = self.conv1(fused_feat)
        x = self.conv2(x)
        x = self.conv3(x)  # Output shape: (B, 3, H_feat, W_feat)

        # 3. 上采样至原图大小以便计算 MSE Loss
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

        return x


class DisentangleLoss(nn.Module):
    """
    解耦相关的损失函数集合
    包含:
    1. L_diff: 差异性损失 (Cosine Similarity)
    2. L_recon: 重建损失 (MSE)
    """

    def __init__(self):
        super(DisentangleLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, feat_inv, feat_spec, recon_img, input_img):
        """
        Args:
            feat_inv: 域不变特征
            feat_spec: 域特定特征
            recon_img: 重建模块生成的图像
            input_img: 原始输入图像 (Target Domain)
        Returns:
            loss_diff, loss_recon
        """
        # --- 1. L_diff (Equation 1) ---
        # L_diff = Mean(|CosineSimilarity(F_inv, F_spec)|)

        # 展平特征以便计算向量相似度: (B, C, H, W) -> (B, C, N)
        b, c, h, w = feat_inv.size()
        f_inv_flat = feat_inv.view(b, c, -1)
        f_spec_flat = feat_spec.view(b, c, -1)

        # 计算余弦相似度 (沿 Channel 维度)
        # dim=1 表示计算通道间的相似度向量
        cos_sim = F.cosine_similarity(f_inv_flat, f_spec_flat, dim=1)

        # 取绝对值并求均值
        loss_diff = torch.mean(torch.abs(cos_sim))

        # --- 2. L_recon (Equation 2) ---
        # L_recon = (1/3N) * Sum((X_hat - X)^2) -> 其实就是 MSE Loss
        loss_recon = self.mse_loss(recon_img, input_img)

        return loss_diff, loss_recon


if __name__ == '__main__':
    # 测试代码
    B, C, H, W = 2, 512, 16, 16  # 假设 bottleneck 特征大小
    H_img, W_img = 512, 512  # 假设原图大小

    # 模拟数据
    feat_inv = torch.randn(B, C, H, W)
    feat_spec = torch.randn(B, C, H, W)
    input_img = torch.randn(B, 3, H_img, W_img)

    # 实例化模块
    reconstructor = ReconstructionModule(in_channels=C)
    loss_func = DisentangleLoss()

    # 前向传播
    recon_img = reconstructor(feat_inv, feat_spec, target_size=(H_img, W_img))

    # 计算损失
    l_diff, l_recon = loss_func(feat_inv, feat_spec, recon_img, input_img)

    print(f"Reconstructed Image Shape: {recon_img.shape}")  # 预期: [2, 3, 512, 512]
    print(f"L_diff: {l_diff.item()}")
    print(f"L_recon: {l_recon.item()}")