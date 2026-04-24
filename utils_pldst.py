import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegratedModel(nn.Module):
    """
    模型包装器：将 Encoder 和 Decoder 视为一个整体。
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, return_feature=True):
        # 动量模型不需要 GRL (alpha=0.0)
        feats_list, _ = self.encoder(x, alpha=0.0)
        logits = self.decoder(feats_list)

        if return_feature:
            return logits, feats_list
        return logits


class PseudoLabelDenoisingSelfTraining:
    def __init__(self, model, momentum_model, momentum=0.99, num_features=512):
        """
        :param model: 主网络 (IntegratedModel)
        :param momentum_model: 动量网络 (IntegratedModel)
        :param momentum: EMA 更新系数
        :param num_features: 特征通道数 (DLinkNet center block 输出是 512)
        """
        self.model = model
        self.momentum_model = momentum_model
        self.momentum = momentum

        # 初始化动量模型
        self.update_momentum_model(beta=0.0)

        # 初始化类特征原型 (Prototype)
        # 使用 register_buffer 确保它能被保存到 state_dict，但不会被优化器更新
        self.model.register_buffer('prototype', torch.zeros(num_features, 1))
        # 标志位：是否已经初始化过原型
        self.has_init_prototype = False

    @torch.no_grad()
    def update_momentum_model(self, beta=None):
        if beta is None:
            beta = self.momentum
        for param_q, param_k in zip(self.model.parameters(), self.momentum_model.parameters()):
            param_k.data = param_k.data * beta + param_q.data * (1. - beta)

    def _extract_feature(self, features):
        """
        [核心修复] 统一处理特征提取逻辑
        如果是列表，取最后一个 (Center Block 特征); 如果是 Tensor，直接用。
        """
        if isinstance(features, (list, tuple)):
            return features[-1]  # DLinkNet: [f1, f2, f3, center] -> center
        return features

    def get_initial_pseudo_label(self, target_img):
        """ 获取主模型的初步预测 """
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(target_img)
            probs = torch.sigmoid(logits)
            initial_pseudo_label = (probs >= 0.5).float()
        return probs, initial_pseudo_label

    def update_prototype(self, features, pseudo_label, alpha=0.9):
        """ 更新特征原型 """
        # 1. 提取并归一化特征
        features = self._extract_feature(features)  # [B, 512, 16, 16]
        # [关键修复] L2 归一化，防止距离计算数值爆炸
        features = F.normalize(features, p=2, dim=1)

        batch_size, feat_dim, h, w = features.size()

        # 2. 对齐伪标签尺寸 (512 -> 16)
        if pseudo_label.shape[2:] != (h, w):
            pseudo_label = F.interpolate(pseudo_label, size=(h, w), mode='nearest')

        # 3. 展平
        features = features.view(batch_size, feat_dim, -1)  # [B, C, N]
        pseudo_label = pseudo_label.view(batch_size, 1, -1)  # [B, 1, N]

        # 4. 计算前景 Mask
        mask = (pseudo_label == 1).float()

        if mask.sum() > 0:
            # 计算当前 Batch 的特征中心
            # sum(features * mask) / count
            current_prototype = (features * mask).sum(dim=(0, 2)) / (mask.sum() + 1e-6)
            current_prototype = current_prototype.view(-1, 1)  # [C, 1]
            # 再次归一化原型，保持在单位球面上
            current_prototype = F.normalize(current_prototype, p=2, dim=0)

            if not self.has_init_prototype:
                self.model.prototype.copy_(current_prototype)
                self.has_init_prototype = True
            else:
                # 动量更新
                new_proto = alpha * self.model.prototype + (1 - alpha) * current_prototype
                new_proto = F.normalize(new_proto, p=2, dim=0)  # 保持单位长度
                self.model.prototype.copy_(new_proto)

    def denoise_pseudo_label(self, target_img):
        """ 生成去噪后的伪标签 """
        # 1. 获取基础概率
        probs, _ = self.get_initial_pseudo_label(target_img)

        # 2. 动量模型提取特征
        self.momentum_model.eval()
        with torch.no_grad():
            _, features_m = self.momentum_model(target_img)

        # 3. 提取特征
        features_m = self._extract_feature(features_m)

        # 4. 如果还没初始化原型，先强制初始化
        if not self.has_init_prototype:
            temp_label = (probs >= 0.5).float()
            self.update_prototype(features_m, temp_label, alpha=0.0)

        # 5. 计算距离权重 (核心去噪步骤)
        # [关键修复] 必须先归一化特征，否则 exp(-dist) 会因距离过大变成 0
        norm_features = F.normalize(features_m, p=2, dim=1)  # [B, 512, 16, 16]

        # 原型扩充维度: [512, 1] -> [1, 512, 1, 1]
        proto_expand = self.model.prototype.view(1, -1, 1, 1)

        # 计算欧氏距离 (由于都归一化了，距离范围在 0~2 之间，非常稳定)
        dist = torch.norm(norm_features - proto_expand, p=2, dim=1, keepdim=True)

        # 距离越小(特征越像)，权重越大
        weight_map = torch.exp(-dist)

        # 6. 上采样权重图 (16 -> 512)
        if weight_map.shape[2:] != probs.shape[2:]:
            weight_map = F.interpolate(weight_map, size=probs.shape[2:], mode='bilinear', align_corners=True)

        # 7. 修正概率并生成标签
        weighted_probs = weight_map * probs
        clean_pseudo_label = (weighted_probs >= 0.5).float()

        # 8. 顺便更新原型
        self.update_prototype(features_m, clean_pseudo_label)

        # [关键] 必须 detach，防止梯度回传到伪标签生成过程
        return clean_pseudo_label.detach()