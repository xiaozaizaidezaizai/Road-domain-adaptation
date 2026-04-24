import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss 用于分割任务，特别适合处理正负样本不平衡（如道路提取）
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, 1, H, W) logits
        # target: (B, 1, H, W) 0 or 1

        pred = torch.sigmoid(pred)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()

        loss = 1 - ((2. * intersection + self.smooth) /
                    (pred_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class SegmentationLoss(nn.Module):
    """
    1. 源域分割损失 (L_seg)
    通常结合 BCEWithLogitsLoss 和 DiceLoss 以获得最佳效果。
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(SegmentationLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice(pred, target)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class AdversarialLoss(nn.Module):
    """
    2. 域不变对抗损失 (L_adv)
    用于训练判别器和生成器（通过 GRL）。
    在 GRL 架构下，我们只需要计算二元分类误差 (BCE)。
    """

    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds_list, is_source):
        """
        Args:
            preds_list: 一个列表，包含多个尺度的判别器输出 [d2, d3, d4]
            is_source: bool, 标记当前输入是否来自源域
                       True -> Label 1
                       False -> Label 0
        """
        loss = 0.0
        target_val = 1.0 if is_source else 0.0

        for pred in preds_list:
            # 创建与预测图形状相同的标签张量
            target = torch.full_like(pred, target_val)
            loss += self.bce(pred, target)

        # 返回平均损失
        return loss / len(preds_list)


        # # 2. [关键修改] 直接取出 d4 (列表的最后一个元素)
        # # 假设 preds_list = [pred_d2, pred_d3, pred_d4]
        # pred_d4 = preds_list[-1]
        #
        # # 3. 创建标签并计算 Loss
        # target = torch.full_like(pred_d4, target_val)
        # loss = self.bce(pred_d4, target)
        #
        # return loss

class AdversarialLoss_st(nn.Module):
    def __init__(self):
        super(AdversarialLoss_st, self).__init__()
        # [防爆炸修复] 必须使用 mean
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, preds_list, is_source):
        loss = 0.0
        target_val = 1.0 if is_source else 0.0

        # 支持单个 Tensor 或 List 输入
        if isinstance(preds_list, list):
            for pred in preds_list:
                target = torch.full_like(pred, target_val)
                loss += self.bce(pred, target)
            return loss / len(preds_list)
        else:
            target = torch.full_like(preds_list, target_val)
            return self.bce(preds_list, target)


class DisentangleLoss(nn.Module):
    """
    包含:
    3. 特征差异损失 (L_diff)
    4. 重建损失 (L_recon)
    """

    def __init__(self):
        super(DisentangleLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, feat_inv, feat_spec, recon_img, input_img):
        """
        Args:
            feat_inv: 域不变特征 (B, C, H, W)
            feat_spec: 域特定特征 (B, C, H, W)
            recon_img: 重建后的图像 (B, 3, H_img, W_img)
            input_img: 原始输入图像 (B, 3, H_img, W_img)
        Returns:
            loss_diff, loss_recon
        """
        # --- L_diff 计算 ---
        # 展平特征: (B, C, H, W) -> (B, C, N)
        b, c, h, w = feat_inv.size()
        f_inv_flat = feat_inv.view(b, c, -1)
        f_spec_flat = feat_spec.view(b, c, -1)

        # 计算通道间的余弦相似度 (dim=1)
        # 我们希望它们不相似(正交)，即 cosine 接近 0
        cos_sim = F.cosine_similarity(f_inv_flat, f_spec_flat, dim=1)
        loss_diff = torch.mean(torch.abs(cos_sim))

        # --- L_recon 计算 ---
        # 简单的 MSE 损失
        loss_recon = self.mse(recon_img, input_img)

        return loss_diff, loss_recon


class TEMLoss(nn.Module):
    """
    5. 目标增强/对比损失 (L_tem)
    让融合特征的预测 (Student) 逼近 域不变特征的预测 (Teacher/Pseudo-label)
    """

    def __init__(self, threshold=0.7):
        super(TEMLoss, self).__init__()
        self.threshold = threshold
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_fused, pred_inv):
        # 1. 两个都转成概率
        prob_fused = torch.sigmoid(pred_fused)

        with torch.no_grad():
            prob_inv = torch.sigmoid(pred_inv)  # Teacher 的软标签

        # 2. 直接计算 MSE 损失 (让学生逼近老师的概率)
        # 或者继续用 BCE，但 target 传入 float 概率
        loss = F.binary_cross_entropy(prob_fused, prob_inv)

        return loss


# 简单的测试代码，用于验证尺寸和运行是否报错
if __name__ == '__main__':
    # 模拟数据
    B, C, H, W = 2, 64, 32, 32
    pred = torch.randn(B, 1, H, W)
    target = torch.randint(0, 2, (B, 1, H, W)).float()

    # 1. 测试 Seg Loss
    criterion_seg = SegmentationLoss()
    print(f"L_seg: {criterion_seg(pred, target).item()}")

    # 2. 测试 Adv Loss
    criterion_adv = AdversarialLoss()
    preds_list = [torch.randn(B, 1, 16, 16), torch.randn(B, 1, 8, 8)]
    print(f"L_adv (Source): {criterion_adv(preds_list, is_source=True).item()}")
    print(f"L_adv (Target): {criterion_adv(preds_list, is_source=False).item()}")

    # 3. 测试 Disentangle Loss
    criterion_dis = DisentangleLoss()
    f_inv = torch.randn(B, C, H, W)
    f_spec = torch.randn(B, C, H, W)
    rec_img = torch.randn(B, 3, 128, 128)
    in_img = torch.randn(B, 3, 128, 128)
    l_diff, l_recon = criterion_dis(f_inv, f_spec, rec_img, in_img)
    print(f"L_diff: {l_diff.item()}, L_recon: {l_recon.item()}")

    # 4. 测试 TEM Loss
    criterion_tem = TEMLoss()
    pred_fused = torch.randn(B, 1, H, W)
    pred_inv = torch.randn(B, 1, H, W)
    print(f"L_tem: {criterion_tem(pred_fused, pred_inv).item()}")