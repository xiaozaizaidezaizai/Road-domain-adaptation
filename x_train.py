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
from model.Reconstruction import ReconstructionModule, DisentangleLoss
from model.losses import TEMLoss, AdversarialLoss, SegmentationLoss
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

# 损失函数权重
LAMBDA_ADV = 0.01  # 保留对抗权重
LAMBDA_DIFF = 0.0  # [消融] 设为0
LAMBDA_RECON = 0.0  # [消融] 设为0
LAMBDA_TEM = 0.0  # [消融] 设为0

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
    enc_invariant = DLinkNetEncoderWithDIFA(pretrained=True).to(DEVICE)
    # [消融] 即使不训练，为了不破坏 Evaluator 的接口，我们还是初始化它，但不放入优化器
    enc_specific = ResNet34Encoder(pretrained=True).to(DEVICE)

    decoder = DLinkNetDecoder(num_classes=1).to(DEVICE)
    # [消融] 重建模块初始化但不使用
    reconstruction_mod = ReconstructionModule(in_channels=512).to(DEVICE)

    # --------------------------
    # 3. 优化器 & 评估器
    # --------------------------
    # [关键修改]：优化器只更新 域不变编码器 和 解码器
    # enc_specific 和 reconstruction_mod 不参与更新
    params = itertools.chain(
        enc_invariant.parameters(),
        decoder.parameters()
    )
    optimizer = optim.Adam(params, lr=LEARNING_RATE, betas=(0.9, 0.99))

    evaluator = Evaluator(enc_invariant, enc_specific, decoder, DEVICE)

    criterion_seg = SegmentationLoss()
    criterion_adv = AdversarialLoss()
    # criterion_dis 和 criterion_tem 这里就不需要了

    # 记录历史
    best_target_iou = 0.0
    history = {
        'epoch': [], 'train_loss': [],
        'target_iou_inv': [], 'target_iou_fused': [],
        'target_f1_inv': [], 'target_f1_fused': [],
        'source_iou_inv': []
    }

    print(f"Start training on {DEVICE} (Ablation: Only Seg + Adv)...")

    for epoch in range(NUM_EPOCHS):
        # --- 训练阶段 ---
        enc_invariant.train()
        decoder.train()

        # [消融] 这些模块设为 eval 模式或者干脆不管
        enc_specific.eval()
        reconstruction_mod.eval()

        epoch_loss = 0.0
        steps = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch")

        for i, (img_s, mask_s, img_t) in enumerate(pbar):
            img_s, mask_s, img_t = img_s.to(DEVICE), mask_s.to(DEVICE), img_t.to(DEVICE)

            # GRL Alpha 动态调整
            p = float(i + epoch * steps) / (NUM_EPOCHS * steps)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            # =================== Forward ===================

            # 1. 源域流 (Source Flow)
            feats_s, dom_preds_s = enc_invariant(img_s, alpha=alpha)
            pred_s = decoder(feats_s)
            loss_seg_s = criterion_seg(pred_s, mask_s)

            # --- 预热判断 ---
            WARMUP_EPOCHS = 0

            if epoch < WARMUP_EPOCHS:
                # 预热阶段：只训练源域分割
                total_loss = loss_seg_s
                loss_adv_total = torch.tensor(0.0)

            else:
                # [消融实验核心修改]
                # 正常 UDA 阶段：只计算对抗损失

                # 2. 目标域流 (Target Flow - Only Adversarial)
                loss_adv_s = criterion_adv(dom_preds_s, is_source=True)
                feats_inv_t, dom_preds_t = enc_invariant(img_t, alpha=alpha)
                loss_adv_t = criterion_adv(dom_preds_t, is_source=False)

                loss_adv_total = (loss_adv_s + loss_adv_t) * 0.5

                # 3. [已移除] 特定编码器、解耦、重建、TEM
                # 为了保持代码简洁，直接注释掉或不写
                # loss_diff = 0
                # loss_recon = 0
                # loss_tem = 0

                # 4. 总损失 (只包含 Seg + Adv)
                total_loss = loss_seg_s + LAMBDA_ADV * loss_adv_total

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            pbar.set_postfix({
                'Loss': f"{total_loss.item():.3f}",
                'Seg': f"{loss_seg_s.item():.3f}",
                'Adv': f"{loss_adv_total.item():.3f}"
                # TEM 不显示了，因为是 0
            })

        avg_loss = epoch_loss / steps
        print(f"Epoch {epoch + 1} Finished. Avg Loss: {avg_loss:.4f}")

        # 保存最新的 Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': enc_invariant.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, os.path.join(SAVE_DIR, 'latest_checkpoint.pth'))

        # --- 评估阶段 ---
        if (epoch + 1) % EVAL_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f"\n[Eval] Epoch {epoch + 1} Evaluation...")

            metrics_t = evaluator.evaluate(target_val_loader, desc="Eval Target")
            iou_t_inv = metrics_t['Inv']['IoU']
            # iou_t_fused 在这里没有意义，因为 SpecificEncoder 没训练，是随机权重

            print(f" >> [Target Inv]   IoU: {iou_t_inv:.4f} (Key Metric for Ablation)")

            metrics_s = evaluator.evaluate(source_val_loader, desc="Eval Source")
            iou_s_inv = metrics_s['Inv']['IoU']
            print(f" >> [Source Inv]   IoU: {iou_s_inv:.4f}")

            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_loss)
            history['target_iou_inv'].append(iou_t_inv)
            history['source_iou_inv'].append(iou_s_inv)

            with open(os.path.join(SAVE_DIR, 'metrics.json'), 'w') as f:
                json.dump(history, f, indent=4)

            # [关键修改] 消融实验只看 Inv 分支，所以用 Inv IoU 保存最佳模型
            if iou_t_inv > best_target_iou:
                best_target_iou = iou_t_inv
                save_path = os.path.join(SAVE_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'enc_invariant': enc_invariant.state_dict(),
                    'decoder': decoder.state_dict(),
                    'best_iou': best_target_iou
                }, save_path)
                print(f" !! New Best Model Saved (Inv IoU: {best_target_iou:.4f})")


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 可以在文件夹名字里加上 ablation 方便区分
    SAVE_DIR = f'./checkpoints/ablation_adv_only_{TIME_STR}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    train()