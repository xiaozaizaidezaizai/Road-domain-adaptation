import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# 导入你的模型和数据集定义
from model.DlinkNet_Encoder import DLinkNetEncoderWithDIFA, DLinkNetDecoder
from Mass_DADataset import RoadUDADataset

# --------------------------
# 配置参数
# --------------------------
SOURCE_ROOT = r"D:\JinWenBo\spacenet"
TARGET_ROOT = r"C:\Users\Administrator\Desktop\RoadData\Massachusetts"

# 修改为你存放 best_model.pth 的实际文件夹路径
SAVE_DIR = r"./checkpoints/MASS_ablation_adv_mix3_zhongjianyu_20260127_215924"
MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
OUTPUT_DIR = os.path.join(SAVE_DIR, 'vis_results') # 改个名字，叫可视化结果

CROP_SIZE = 512
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------
# 简易指标计算器
# --------------------------
class RoadMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred, target):
        # pred, target: [B, 1, H, W] (0 or 1)
        pred = pred.cpu().numpy().astype(bool)
        target = target.cpu().numpy().astype(bool)

        self.tp += np.sum(pred & target)
        self.fp += np.sum(pred & ~target)
        self.fn += np.sum(~pred & target)
        self.tn += np.sum(~pred & ~target)

    def get_score(self):
        smooth = 1e-6
        iou = self.tp / (self.tp + self.fp + self.fn + smooth)
        precision = self.tp / (self.tp + self.fp + smooth)
        recall = self.tp / (self.tp + self.fn + smooth)
        f1 = 2 * precision * recall / (precision + recall + smooth)
        acc = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + smooth)
        return iou, f1, precision, recall, acc


# --------------------------
# [新增] 可视化拼接函数
# --------------------------
def save_visualization(img_tensor, mask_tensor, pred_tensor, save_path):
    """
    将 原图、标签(GT)、预测(Pred) 拼接成一张长图保存
    img_tensor: [1, 3, H, W] (Normalized)
    mask_tensor: [1, 1, H, W] (0/1)
    pred_tensor: [1, 1, H, W] (0/1)
    """
    # 1. 处理原图 (反归一化，转为 HWC, 0-255)
    img = img_tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0) # H, W, 3
    # 简单反归一化：将数据拉伸到 0-1 之间以便显示 (防止因为 mean/std 导致图像黑乎乎)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img_uint8 = (img * 255).astype(np.uint8)

    # 2. 处理标签 (GT)
    gt = mask_tensor.squeeze().cpu().detach().numpy() # H, W
    gt_uint8 = (gt * 255).astype(np.uint8)
    # 转为 3通道 灰度图，方便拼接
    gt_rgb = np.stack([gt_uint8, gt_uint8, gt_uint8], axis=2)

    # 3. 处理预测 (Pred)
    pred = pred_tensor.squeeze().cpu().detach().numpy() # H, W
    pred_uint8 = (pred * 255).astype(np.uint8)
    pred_rgb = np.stack([pred_uint8, pred_uint8, pred_uint8], axis=2)

    # 4. 拼接：[原图 | 标签 | 预测]
    # 如果你想上下拼接，用 vstack；这里用 hstack (左右拼接)
    combined = np.hstack([img_uint8, gt_rgb, pred_rgb])

    # 5. 保存
    Image.fromarray(combined).save(save_path)


# --------------------------
# 主运行函数
# --------------------------
def main_inference():
    # 1. 准备
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"Start processing... Visualization will be saved to {OUTPUT_DIR}")
    print("Format: [ Original Image | Ground Truth | Prediction ]")

    metric = RoadMetric()

    # 2. 数据
    val_dataset = RoadUDADataset(
        source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
        source_list_name="train.txt", target_list_name="train_cleaned.txt",
        crop_size=CROP_SIZE, mode='val'
    )
    # num_workers=0 避免多进程报错
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 模型
    enc = DLinkNetEncoderWithDIFA(pretrained=False).to(DEVICE)
    dec = DLinkNetDecoder(num_classes=1).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        enc.load_state_dict(ckpt['enc_invariant'])
        dec.load_state_dict(ckpt['decoder'])
        print("Model loaded successfully.")
    else:
        print("Checkpoint not found!")
        return

    enc.eval()
    dec.eval()
    # 4. 推理
    print("Running evaluation...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
            # 兼容不同的 dataset 返回
            if len(data) == 3:
                img, mask, name = data
                filename = name[0]
            else:
                img, mask = data
                filename = f"pred_{i}.png"

            # 去掉后缀，准备加新后缀
            raw_name = os.path.splitext(filename)[0]

            # ==================================================
            # [核心修改] 防止覆盖！给文件名加上切片后缀
            # ==================================================
            # 因为数据是按顺序加载的 (shuffle=False)
            # i % 4 的结果分别是 0, 1, 2, 3，对应左上、右上、左下、右下
            patch_idx = i % 4
            filename = f"{raw_name}_patch_{patch_idx}.png"
            # ==================================================

            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            # Forward
            f, _ = enc(img, alpha=0.0)
            p = dec(f)

            prob = torch.sigmoid(p)
            pred_bin = (prob > 0.5).float()

            # Metric
            metric.update(pred_bin, mask)

            # 保存
            save_path = os.path.join(OUTPUT_DIR, filename)
            save_visualization(img, mask, pred_bin, save_path)
    # # 4. 推理
    # print("Running evaluation...")
    # with torch.no_grad():
    #     for i, data in enumerate(tqdm(val_loader)):
    #         # 兼容不同的 dataset 返回
    #         if len(data) == 3:
    #             img, mask, name = data
    #             filename = name[0]
    #         else:
    #             img, mask = data
    #             filename = f"pred_{i}.png"
    #
    #         if not filename.endswith('.png'):
    #             filename = os.path.splitext(filename)[0] + ".png"
    #
    #         img = img.to(DEVICE)
    #         mask = mask.to(DEVICE)
    #
    #         # Forward
    #         f, _ = enc(img, alpha=0.0)
    #         p = dec(f)
    #
    #         prob = torch.sigmoid(p)
    #         pred_bin = (prob > 0.5).float()
    #
    #         # Metric
    #         metric.update(pred_bin, mask)
    #
    #         # --- [核心修改] 调用可视化拼接函数 ---
    #         save_path = os.path.join(OUTPUT_DIR, filename)
    #         save_visualization(img, mask, pred_bin, save_path)
    #         # --------------------------------

    # 5. 结果
    iou, f1, pre, rec, acc = metric.get_score()
    res_str = (f"IoU: {iou:.4f} | F1: {f1:.4f} | Pre: {pre:.4f} | Rec: {rec:.4f} | Acc: {acc:.4f}")
    print("\n" + "=" * 30)
    print(res_str)
    print("=" * 30)

    with open(os.path.join(SAVE_DIR, 'eval_result.txt'), 'w') as f:
        f.write(res_str)


if __name__ == '__main__':
    main_inference()