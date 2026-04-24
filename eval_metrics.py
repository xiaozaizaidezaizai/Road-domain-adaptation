import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class Evaluator:
    def __init__(self, enc_invariant, enc_specific, decoder, device):
        """
        Args:
            enc_invariant: 域不变编码器
            enc_specific: 域特定编码器 (新增)
            decoder: 共享解码器
            device: 设备
        """
        self.enc_invariant = enc_invariant
        self.enc_specific = enc_specific
        self.decoder = decoder
        self.device = device

    def _update_confusion_matrix(self, pred, label, confusion_matrix):
        pred = pred.view(-1).byte()
        label = label.view(-1).byte()

        TP = ((pred == 1) & (label == 1)).sum().item()
        FP = ((pred == 1) & (label == 0)).sum().item()
        FN = ((pred == 0) & (label == 1)).sum().item()
        TN = ((pred == 0) & (label == 0)).sum().item()

        confusion_matrix['TP'] += TP
        confusion_matrix['FP'] += FP
        confusion_matrix['FN'] += FN
        confusion_matrix['TN'] += TN

    def _calculate_metrics(self, cm):
        """辅助函数：根据混淆矩阵计算指标"""
        TP, FP, FN = cm['TP'], cm['FP'], cm['FN']
        smooth = 1e-6

        iou = TP / (TP + FP + FN + smooth)
        precision = TP / (TP + FP + smooth)
        recall = TP / (TP + FN + smooth)
        f1 = 2 * (precision * recall) / (precision + recall + smooth)

        return {'IoU': iou, 'F1': f1, 'Precision': precision, 'Recall': recall}

    def evaluate(self, data_loader, desc="Evaluating"):
        self.enc_invariant.eval()
        if self.enc_specific is not None:  # [修改] 只有当它存在时才调用 eval
            self.enc_specific.eval()
        self.decoder.eval()

        # 维护两个混淆矩阵
        cm_inv = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}  # 仅域不变特征
        cm_fused = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}  # 融合特征 (不变 + 特定)

        with torch.no_grad():
            for img, mask, _ in tqdm(data_loader, desc=desc, leave=True):
                img = img.to(self.device)
                mask = mask.to(self.device)

                # 1. 提取特征
                # 不变特征 (alpha=0 关闭 GRL)
                feat_inv, _ = self.enc_invariant(img, alpha=0.0)
                # 特定特征
                feat_spec = self.enc_specific(img)

                # 2. 预测 A: 仅使用域不变特征 (Baseline)
                pred_inv_logits = self.decoder(feat_inv)

                # 3. 预测 B: 使用融合特征 (Ours)
                # 构造融合特征列表 (假设在 Layer 4 融合)
                # 注意：ResNet Encoder 返回的是列表 [f1, f2, f3, f4]
                feat_fused = []
                for f_inv, f_spec in zip(feat_inv, feat_spec):
                    feat_fused.append(f_inv + f_spec)

                pred_fused_logits = self.decoder(feat_fused)

                # 4. 后处理
                mask_inv = (torch.sigmoid(pred_inv_logits) > 0.5).float()
                mask_fused = (torch.sigmoid(pred_fused_logits) > 0.5).float()

                # 5. 更新矩阵
                self._update_confusion_matrix(mask_inv, mask, cm_inv)
                self._update_confusion_matrix(mask_fused, mask, cm_fused)

        # 计算最终指标
        metrics_inv = self._calculate_metrics(cm_inv)
        metrics_fused = self._calculate_metrics(cm_fused)

        return {
            'Inv': metrics_inv,
            'Fused': metrics_fused
        }