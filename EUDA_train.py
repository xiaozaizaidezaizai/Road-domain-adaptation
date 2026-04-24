import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import itertools
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

# 导入自定义模块
from EDUA_eval_metrics import Evaluator
from model.DANnet import DAN_Net_Decoder, ResNet34Encoder
from model.losses import AdversarialLoss, SegmentationLoss
from DADataset import RoadUDADataset
from model.DlinkNet_Encoder import DLinkNetEncoderWithDIFA, DLinkNetDecoder

# --------------------------
# 配置参数 (Configuration)
# --------------------------
SOURCE_ROOT = r"D:\JinWenBo\spacenet"
TARGET_ROOT = r"C:\Users\Administrator\Desktop\RoadData\deepglobe"

# 训练超参数
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
NUM_EPOCHS = 200
EVAL_INTERVAL = 5
CROP_SIZE = 512

# 损失权重与 EUDA-PLR 参数
LAMBDA_ADV = 0.01
LAMBDA_PSEUDO = 0.5
TOP_K_PERCENT = 0.2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = ""


# --------------------------
# 辅助函数
# --------------------------

def update_ema(student_model, teacher_model, alpha=0.99):
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data


def get_topk_pseudo_label(teacher_prob, k_percent=0.2):
    B, C, H, W = teacher_prob.shape
    pseudo_mask = torch.zeros_like(teacher_prob).to(teacher_prob.device)
    for b in range(B):
        prob = teacher_prob[b].view(-1)
        k_val = int(prob.numel() * k_percent)
        if k_val > 0:
            topk_val, _ = torch.topk(prob, k_val)
            threshold = topk_val[-1]
            mask = (prob >= threshold).float().view(1, H, W)
            pseudo_mask[b] = mask
    return pseudo_mask


# [新增] 保存日志到 JSON 的函数
def save_logs_to_json(log_data, file_path):
    # 如果文件存在，先读取旧数据，再追加；或者直接保存整个列表
    # 这里我们采用保存完整列表的方式，每次覆盖写入，防止格式错误
    with open(file_path, 'w') as f:
        json.dump(log_data, f, indent=4)


# --------------------------
# 训练主循环
# --------------------------

def train():
    global SAVE_DIR
    print(f"Model will be saved to: {SAVE_DIR}")

    # ================= 1. 数据加载 =================
    print(f"Loading data...")

    # 训练集 (混合源域和目标域)
    train_dataset = RoadUDADataset(source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
                                   source_list_name="train.txt", target_list_name="train.txt",
                                   crop_size=CROP_SIZE, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)

    # [新增] 源域验证集 (SpaceNet)
    # 假设源域也有 train.txt 或 val.txt，这里为了演示使用 train.txt，建议改为 val.txt
    source_val_dataset = RoadUDADataset(source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
                                        source_list_name="train.txt", target_list_name="train.txt",
                                        crop_size=CROP_SIZE, mode='val_source')
    # 注意：需确保你的 Dataset 类在 mode='val_source' 时只返回源域图片和标签
    # 如果 Dataset 不支持，通常 mode='val' 时是根据 source_root 还是 target_root 来区分的，
    # 或者你需要实例化两个不同的 Dataset 对象，分别指向不同的 root。

    # 为了保险起见，这里假设 Dataset 在 validation 时通常只看 target。
    # 如果你的 Dataset 类没写分源域验证的逻辑，你可能需要手动指定验证列表。
    # 这里暂且复用 Dataset，假设它能正确处理。
    source_val_loader = DataLoader(source_val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 目标域验证集 (DeepGlobe)
    target_val_dataset = RoadUDADataset(source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
                                        source_list_name="train.txt", target_list_name="train.txt",
                                        crop_size=CROP_SIZE, mode='val')
    target_val_loader = DataLoader(target_val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # ================= 2. 模型初始化 =================
    print("Initializing Student models...")
    enc_invariant = DLinkNetEncoderWithDIFA(pretrained=True).to(DEVICE)
    decoder = DLinkNetDecoder(num_classes=1).to(DEVICE)

    print("Initializing Teacher models (EMA)...")
    enc_teacher = DLinkNetEncoderWithDIFA(pretrained=True).to(DEVICE)
    dec_teacher = DLinkNetDecoder(num_classes=1).to(DEVICE)

    # 冻结教师网络
    for param in enc_teacher.parameters(): param.requires_grad = False
    for param in dec_teacher.parameters(): param.requires_grad = False

    # 初始化 EMA
    update_ema(enc_invariant, enc_teacher, alpha=0.0)
    update_ema(decoder, dec_teacher, alpha=0.0)

    # ================= 3. 优化器与评估器 =================
    params = itertools.chain(enc_invariant.parameters(), decoder.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE, betas=(0.9, 0.99))

    # [关键] 传入学生网络进行评估
    evaluator = Evaluator(enc_invariant, decoder, DEVICE)

    criterion_seg = SegmentationLoss()
    criterion_adv = AdversarialLoss()

    # 初始化日志记录列表
    training_history = []
    json_log_path = os.path.join(SAVE_DIR, 'training_log.json')
    best_target_iou = 0.0

    print(f"Start training on {DEVICE}...")

    # ================= 4. Epoch 循环 =================
    for epoch in range(NUM_EPOCHS):
        # --- 训练阶段 ---
        enc_invariant.train()
        decoder.train()
        enc_teacher.eval()
        dec_teacher.eval()

        epoch_loss = 0.0
        steps = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch")

        for i, (img_s, mask_s, img_t) in enumerate(pbar):
            img_s, mask_s, img_t = img_s.to(DEVICE), mask_s.to(DEVICE), img_t.to(DEVICE)

            # GRL Alpha 动态计算
            p = float(i + epoch * steps) / (NUM_EPOCHS * steps)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            # 1. 源域流
            feats_s, dom_preds_s = enc_invariant(img_s, alpha=alpha)
            pred_s = decoder(feats_s)
            loss_seg_s = criterion_seg(pred_s, mask_s)

            # 2. 目标域流 (教师生成伪标签)
            with torch.no_grad():
                feats_t_teacher, _ = enc_teacher(img_t, alpha=0)
                pred_t_teacher = dec_teacher(feats_t_teacher)
                prob_t_teacher = torch.sigmoid(pred_t_teacher)
                pseudo_label_t = get_topk_pseudo_label(prob_t_teacher, k_percent=TOP_K_PERCENT)

            # 3. 目标域流 (学生训练)
            feats_t, dom_preds_t = enc_invariant(img_t, alpha=alpha)
            pred_t = decoder(feats_t)

            # 4. 损失计算
            WARMUP_EPOCHS = 10
            if epoch < WARMUP_EPOCHS:
                total_loss = loss_seg_s
                loss_adv_total = torch.tensor(0.0).to(DEVICE)
                loss_pseudo_t = torch.tensor(0.0).to(DEVICE)
            else:
                loss_adv_s = criterion_adv(dom_preds_s, is_source=True)
                loss_adv_t = criterion_adv(dom_preds_t, is_source=False)
                loss_adv_total = (loss_adv_s + loss_adv_t) * 0.5
                loss_pseudo_t = criterion_seg(pred_t, pseudo_label_t)

                total_loss = loss_seg_s + LAMBDA_ADV * loss_adv_total + LAMBDA_PSEUDO * loss_pseudo_t

            total_loss.backward()
            optimizer.step()

            # 更新教师
            update_ema(enc_invariant, enc_teacher, alpha=0.99)
            update_ema(decoder, dec_teacher, alpha=0.99)

            epoch_loss += total_loss.item()
            pbar.set_postfix(
                {'L_All': f"{total_loss.item():.3f}", 'Mode': 'Pretrain' if epoch < WARMUP_EPOCHS else 'Adapt'})

        avg_loss = epoch_loss / steps
        print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.4f}")

        # 保存最新的 checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': enc_invariant.state_dict(),
            'decoder_state': decoder.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, os.path.join(SAVE_DIR, 'latest_checkpoint.pth'))

        # --- 评估阶段 (同时评估源域和目标域) ---
        if (epoch + 1) % EVAL_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f"\n[Eval] Epoch {epoch + 1}...")

            # 1. 评估源域 (SpaceNet)
            metrics_s = evaluator.evaluate(source_val_loader, desc="Eval Source (SpaceNet)")
            iou_s = metrics_s['Inv']['IoU']
            print(f" >> [Source] IoU: {iou_s:.4f} | F1: {metrics_s['Inv']['F1']:.4f}")

            # 2. 评估目标域 (DeepGlobe)
            metrics_t = evaluator.evaluate(target_val_loader, desc="Eval Target (DeepGlobe)")
            iou_t = metrics_t['Inv']['IoU']
            print(f" >> [Target] IoU: {iou_t:.4f} | F1: {metrics_t['Inv']['F1']:.4f}")

            # --- 保存结果到 JSON ---
            current_log = {
                "epoch": epoch + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "train_loss": avg_loss,
                "source_metrics": metrics_s['Inv'],  # 包含 IoU, F1, Precision, Recall
                "target_metrics": metrics_t['Inv']
            }
            training_history.append(current_log)
            save_logs_to_json(training_history, json_log_path)
            print(f" >> Logs saved to {json_log_path}")

            # --- 保存最佳模型 (依然以 Target IoU 为标准) ---
            if iou_t > best_target_iou:
                best_target_iou = iou_t
                torch.save({
                    'epoch': epoch,
                    'enc_invariant': enc_invariant.state_dict(),
                    'decoder': decoder.state_dict(),
                    'enc_teacher': enc_teacher.state_dict(),
                    'dec_teacher': dec_teacher.state_dict(),
                    'best_iou': best_target_iou,
                    'source_iou': iou_s  # 顺便记录下此刻的源域效果
                }, os.path.join(SAVE_DIR, 'best_model.pth'))
                print(f" !! New Best Model (Target IoU: {best_target_iou:.4f})")


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = f'./checkpoints/EUDA_PLR_Road_{TIME_STR}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    train()