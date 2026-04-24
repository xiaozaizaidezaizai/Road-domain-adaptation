import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# 导入你的模型定义
# 确保 DLinkNet_Encoder.py 在能引用的路径下
from model.DlinkNet_Encoder import DLinkNetEncoderWithDIFA

# ================= 配置 =================
# 1. 图片路径 (换成你想测试的那张图)
IMG_PATH = r"D:\JinWenBo\CHN6-CUG\train\am100003_sat.jpg"

# 2. 权重路径
MODEL_PATH = r"./checkpoints/MASS_ablation_adv_mix3_zhongjianyu_20260127_215924/best_model.pth"

# 3. 保存路径
SAVE_PATH = "heatmap_result.png"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =======================================

# --- 1. 定义 Hook 类用于提取中间特征 ---
class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.features = {}
        self.hooks = []

        # 注册 Hook
        for layer_name in target_layers:
            layer = dict([*self.model.named_modules()])[layer_name]
            self.hooks.append(layer.register_forward_hook(self._get_hook(layer_name)))

    def _get_hook(self, name):
        def hook(model, input, output):
            self.features[name] = output

        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# --- 2. 热力图转换函数 ---
def feature_to_heatmap(feature_tensor, target_size=(512, 512)):
    """
    将 (C, H, W) 的特征图转换为 RGB 热力图
    原理: 对 Channel 维度求均值 -> 归一化 -> 应用伪彩色
    """
    # 1. 压缩通道: (1, C, H, W) -> (H, W)
    # 使用 mean (关注整体) 或 max (关注最强响应) 均可，这里常用 mean
    heatmap = torch.mean(feature_tensor, dim=1).squeeze().cpu().detach().numpy()

    # 2. 归一化到 [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 3. 缩放到目标尺寸 (比如原图大小)
    heatmap = cv2.resize(heatmap, target_size)

    # 4. 转换为 0-255 并应用色彩映射 (JET 也就是 蓝-青-黄-红)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # OpenCV 是 BGR，转为 RGB
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


def main():
    # 1. 加载模型
    print("Loading model...")
    # 注意：我们只需要 Encoder 就可以提取特征了
    enc = DLinkNetEncoderWithDIFA(pretrained=False).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        # 你的权重字典键名可能是 'enc_invariant'
        enc.load_state_dict(ckpt['enc_invariant'])
        print("✅ Weights loaded.")
    else:
        print("❌ Model path not found.")
        return

    enc.eval()

    # 2. 注册 Hook (拦截 ResNet 的 4 个层级)
    # ResNet34 的标准层名通常是: layer1, layer2, layer3, layer4
    target_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    extractor = FeatureExtractor(enc, target_layers)

    # 3. 读取和预处理图片
    raw_img = Image.open(IMG_PATH).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

    # 4. 前向传播 (Hook 会自动捕获特征)
    with torch.no_grad():
        _ = enc(input_tensor)

    # 5. 绘图
    print("Generating heatmaps...")
    plt.figure(figsize=(20, 5))

    # 子图 1: 原图
    plt.subplot(1, 5, 1)
    plt.imshow(raw_img.resize((512, 512)))
    plt.title("Original Image")
    plt.axis('off')

    # 子图 2-5: 四个尺度的特征热力图
    layer_titles = ["1/4 Scale (Layer1)", "1/8 Scale (Layer2)", "1/16 Scale (Layer3)", "1/32 Scale (Layer4)"]

    for i, layer_name in enumerate(target_layers):
        feat = extractor.features[layer_name]  # 获取捕获的特征
        heatmap = feature_to_heatmap(feat, target_size=(512, 512))

        plt.subplot(1, 5, i + 2)
        plt.imshow(heatmap)
        plt.title(layer_titles[i])
        plt.axis('off')

        # 叠加模式 (可选：如果你想看叠加在原图上的效果)
        # alpha = 0.5
        # original_np = np.array(raw_img.resize((512, 512)))
        # overlay = (original_np * (1-alpha) + heatmap * alpha).astype(np.uint8)
        # plt.imshow(overlay)

    # 6. 保存
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"✅ Result saved to {SAVE_PATH}")
    plt.show()

    # 清理 Hook
    extractor.remove_hooks()


if __name__ == '__main__':
    main()