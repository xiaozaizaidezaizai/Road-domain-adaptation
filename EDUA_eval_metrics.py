import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class Evaluator:
    def __init__(self, encoder, decoder, device):
        """
        专为单流架构 (Mean Teacher / Adversarial) 设计的评估器
        Args:
            encoder: 学生网络的编码器 (enc_invariant)
            decoder: 学生网络的解码器 (decoder)
            device: 设备 (cuda/cpu)
        """
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def _update_confusion_matrix(self, pred, label, confusion_matrix):
        """更新混淆矩阵 (TP, FP, FN, TN)"""
        # 展平并转为 byte 类型以节省内存
        pred = pred.view(-1).byte()
        label = label.view(-1).byte()

        # 计算各类数量
        TP = ((pred == 1) & (label == 1)).sum().item()
        FP = ((pred == 1) & (label == 0)).sum().item()
        FN = ((pred == 0) & (label == 1)).sum().item()
        TN = ((pred == 0) & (label == 0)).sum().item()

        # 累加到总字典中
        confusion_matrix['TP'] += TP
        confusion_matrix['FP'] += FP
        confusion_matrix['FN'] += FN
        confusion_matrix['TN'] += TN

    def _calculate_metrics(self, cm):
        """根据累计的混淆矩阵计算 IoU, F1 等指标"""
        TP, FP, FN = cm['TP'], cm['FP'], cm['FN']
        smooth = 1e-6

        # 公式计算
        iou = TP / (TP + FP + FN + smooth)
        precision = TP / (TP + FP + smooth)
        recall = TP / (TP + FN + smooth)
        f1 = 2 * (precision * recall) / (precision + recall + smooth)

        return {'IoU': iou, 'F1': f1, 'Precision': precision, 'Recall': recall}

    def evaluate(self, data_loader, desc="Evaluating"):
        """执行评估循环"""
        self.encoder.eval()
        self.decoder.eval()

        # 初始化混淆矩阵
        cm = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

        with torch.no_grad():
            for img, mask, _ in tqdm(data_loader, desc=desc, leave=True):
                img = img.to(self.device)
                mask = mask.to(self.device)

                # 1. 前向传播 (Forward)
                # alpha=0 关闭梯度反转层带来的影响（虽然 eval 模式下 dropout/bn 已变，但设为 0 更严谨）
                feats, _ = self.encoder(img, alpha=0.0)
                pred_logits = self.decoder(feats)

                # 2. 生成预测掩码 (0 或 1)
                pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()

                # 3. 更新统计
                self._update_confusion_matrix(pred_mask, mask, cm)

        # 4. 计算最终指标
        metrics = self._calculate_metrics(cm)

        # 为了兼容你的 train.py 代码 (lines 256-257)，
        # 我们保持返回字典的结构为 {'Inv': metrics}
        # 这样你的 train.py 里 metrics_t['Inv']['IoU'] 就不会报错了
        return {
            'Inv': metrics
        }