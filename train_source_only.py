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
from eval_metrics import Evaluator
from model.DANnet import DAN_Net_Decoder, ResNet34Encoder
from model.Reconstruction import ReconstructionModule
from model.losses import SegmentationLoss
from DADataset import RoadUDADataset
from model.DlinkNet_Encoder import DLinkNetEncoderWithDIFA, DLinkNetDecoder

# --------------------------
# 配置参数 (Configuration)
# --------------------------
SOURCE_ROOT = r"D:\JinWenBo\spacenet"
TARGET_ROOT = r"C:\Users\Administrator\Desktop\RoadData\deepglobe"

# 训练超参数
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
NUM_EPOCHS = 200
EVAL_INTERVAL = 5
CROP_SIZE = 512

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = ""


def train():
    global SAVE_DIR
    print(f"Model will be saved to: {SAVE_DIR}")

    # --------------------------
    # 1. 数据加载
    # --------------------------
    print(f"Loading data... (Source: {SOURCE_ROOT}, Target: {TARGET_ROOT})")

    train_dataset = RoadUDADataset(
        source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
        source_list_name="train.txt", target_list_name="train.txt",
        crop_size=CROP_SIZE, mode='train'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True
    )

    target_val_dataset = RoadUDADataset(
        source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
        source_list_name="train.txt", target_list_name="train.txt",
        crop_size=CROP_SIZE, mode='val'
    )
    target_val_loader = DataLoader(target_val_dataset, batch_size=1, shuffle=False, num_workers=2)

    source_val_dataset = RoadUDADataset(
        source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
        source_list_name="train.txt", target_list_name="train.txt",
        crop_size=CROP_SIZE, mode='val_source'
    )
    source_val_loader = DataLoader(source_val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # --------------------------
    # 2. 模型初始化
    # --------------------------
    print("Initializing models...")
    # 依然使用 DLinkNet 结构
    enc_invariant = DLinkNetEncoderWithDIFA(pretrained=True).to(DEVICE)
    decoder = DLinkNetDecoder(num_classes=1).to(DEVICE)

    # [关键修复]：虽然是 Source Only，为了让 Evaluator 不报错，
    # 我们必须初始化一个 enc_specific 占位。
    # 为了省显存和加载时间，pretrained 可以设为 False
    enc_specific = ResNet34Encoder(pretrained=False).to(DEVICE)

    # 把它锁死在 eval 模式，防止它产生任何梯度或更新 Batch Norm
    enc_specific.eval()
    for param in enc_specific.parameters():
        param.requires_grad = False

    # --------------------------
    # 3. 优化器 & 评估器
    # --------------------------
    # [关键]：优化器参数列表里，绝对不能包含 enc_specific
    params = itertools.chain(
        enc_invariant.parameters(),
        decoder.parameters()
    )
    optimizer = optim.Adam(params, lr=LEARNING_RATE, betas=(0.9, 0.99))

    # 现在传进去的是一个真实存在的模型对象，而不是 None
    evaluator = Evaluator(enc_invariant, enc_specific, decoder, DEVICE)

    criterion_seg = SegmentationLoss()

    # 记录历史
    best_target_iou = 0.0
    history = {
        'epoch': [], 'train_loss': [],
        'target_iou_inv': [], 'source_iou_inv': []
    }

    print(f"Start training on {DEVICE} (Ablation: Source Only)...")

    for epoch in range(NUM_EPOCHS):
        # --- 训练阶段 ---
        enc_invariant.train()
        decoder.train()

        epoch_loss = 0.0
        steps = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch")

        for i, (img_s, mask_s, img_t) in enumerate(pbar):
            img_s, mask_s = img_s.to(DEVICE), mask_s.to(DEVICE)
            # img_t 虽然加载了，但我们直接忽略它

            optimizer.zero_grad()

            # =================== Forward ===================

            # 1. 源域流 (Source Flow)
            # alpha 传 0.0 或任意值都行，因为我们不计算 domain_preds 的损失，所以梯度不会传给判别器
            feats_s, _ = enc_invariant(img_s, alpha=0.0)
            pred_s = decoder(feats_s)

            # 只计算分割损失
            loss_seg_s = criterion_seg(pred_s, mask_s)

            # 2. 总损失
            total_loss = loss_seg_s

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            pbar.set_postfix({
                'Loss': f"{total_loss.item():.3f}",
                'Seg': f"{loss_seg_s.item():.3f}"
            })

        avg_loss = epoch_loss / steps
        print(f"Epoch {epoch + 1} Finished. Avg Loss: {avg_loss:.4f}")

        # 保存最新的 Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': enc_invariant.state_dict(),
            'decoder_state': decoder.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, os.path.join(SAVE_DIR, 'latest_checkpoint.pth'))

        # --- 评估阶段 ---
        if (epoch + 1) % EVAL_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f"\n[Eval] Epoch {epoch + 1} Evaluation...")

            # Evaluator 内部虽然写了 Specific 的逻辑，但如果不传 Specific Encoder 可能会报错
            # 所以我们需要确保 Evaluator 代码能处理 None，或者我们只看 Inv 指标
            # 如果之前的 Evaluator 代码里写死了 self.enc_specific(img)，这里可能会报错
            # 为了保险起见，建议用 try-except 或者确保 Evaluator 逻辑兼容
            # (假设你用的是之前包含 Inv/Fused 分开计算的 Evaluator)

            # 如果 Evaluator 比较复杂，这里我们也可以手动写个简单的评估逻辑，或者修改 Evaluator
            # 这里假设 Evaluator 的 evaluate 方法能跑通 (它会忽略 Fused 部分如果出错的话，或者我们只取 Inv)

            try:
                metrics_t = evaluator.evaluate(target_val_loader, desc="Eval Target")
                iou_t_inv = metrics_t['Inv']['IoU']
                print(f" >> [Target Inv]   IoU: {iou_t_inv:.4f} (Source Only Baseline)")

                metrics_s = evaluator.evaluate(source_val_loader, desc="Eval Source")
                iou_s_inv = metrics_s['Inv']['IoU']
                print(f" >> [Source Inv]   IoU: {iou_s_inv:.4f}")

                history['epoch'].append(epoch + 1)
                history['train_loss'].append(avg_loss)
                history['target_iou_inv'].append(iou_t_inv)
                history['source_iou_inv'].append(iou_s_inv)

                with open(os.path.join(SAVE_DIR, 'metrics.json'), 'w') as f:
                    json.dump(history, f, indent=4)

                if iou_t_inv > best_target_iou:
                    best_target_iou = iou_t_inv
                    save_path = os.path.join(SAVE_DIR, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'enc_invariant': enc_invariant.state_dict(),
                        'decoder': decoder.state_dict(),
                        'best_iou': best_target_iou
                    }, save_path)
                    print(f" !! New Best Model Saved (IoU: {best_target_iou:.4f})")

            except Exception as e:
                print(f"Evaluation failed: {e}")
                print("Tip: Make sure Evaluator handles None for enc_specific or create a dummy one.")
                # 为了防止报错，这里也可以临时实例化一个假的 enc_specific 传进去，只要不训练它就行


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = f'./checkpoints/ablation_source_only_{TIME_STR}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    train()
