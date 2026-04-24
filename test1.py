import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# 导入你的模型和数据集定义
# 确保这些文件路径正确
from model.DlinkNet_Encoder import DLinkNetEncoderWithDIFA, DLinkNetDecoder
from DADataset import RoadUDADataset

# --------------------------
# 配置参数
# --------------------------
SOURCE_ROOT = r"D:\JinWenBo\spacenet"
TARGET_ROOT = r"C:\Users\Administrator\Desktop\RoadData\deepglobe"

# 修改为你存放 best_model.pth 的实际文件夹路径
SAVE_DIR = r"./checkpoints/ablation_adv_mix_zhong_noDlink_20260125_194313"
MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
OUTPUT_DIR = os.path.join(SAVE_DIR, 'test_results_target')

CROP_SIZE = 512
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------
# 简易指标计算器 (不依赖外部库)
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
# 主运行函数 (改名为 main 避免被当成测试)
# --------------------------
def main_inference():
    # 1. 准备
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"Start processing... Save to {OUTPUT_DIR}")

    metric = RoadMetric()

    # 2. 数据
    val_dataset = RoadUDADataset(
        source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
        source_list_name="train.txt", target_list_name="train.txt",
        crop_size=CROP_SIZE, mode='val'
    )
    # num_workers=0 避免多进程报错，稳定第一
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

            # 确保 filename 是字符串且有后缀
            if not filename.endswith('.png'):
                filename = os.path.splitext(filename)[0] + ".png"

            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            # Forward
            f, _ = enc(img, alpha=0.0)
            p = dec(f)

            prob = torch.sigmoid(p)
            pred_bin = (prob > 0.5).float()

            # Metric
            metric.update(pred_bin, mask)

            # Save
            pred_np = pred_bin.squeeze().cpu().numpy()
            Image.fromarray((pred_np * 255).astype(np.uint8)).save(os.path.join(OUTPUT_DIR, filename))

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