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
from model.EncoderDIFA import ResNet34EncoderWithDIFA
from model.Reconstruction import ReconstructionModule, DisentangleLoss
from model.losses import TEMLoss, AdversarialLoss, SegmentationLoss
from DADataset import RoadUDADataset
from model.DlinkNet_Encoder import DLinkNetEncoderWithDIFA, DLinkNetDecoder
# --------------------------
# 配置参数 (Configuration)
# --------------------------
# 路径配置 (请修改为你的实际路径)
SOURCE_ROOT = r"D:\JinWenBo\spacenet"
TARGET_ROOT = r"C:\Users\Administrator\Desktop\RoadData\deepglobe"

# 训练超参数
LEARNING_RATE = 2e-4
BATCH_SIZE = 8  # 根据显存调整，若 OOM 请改为 4 或 2
NUM_EPOCHS = 200  # 总训练轮数
EVAL_INTERVAL = 5  # 每几轮评估一次
CROP_SIZE = 512  # 训练时裁剪大小

# 损失函数权重
LAMBDA_ADV = 0.01
LAMBDA_DIFF = 0.1
LAMBDA_RECON = 0.1
LAMBDA_TEM = 0.5

# 设备与保存路径
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SAVE_DIR = ""

def train():
    global SAVE_DIR
    print(f"Model will be saved to: {SAVE_DIR}")
    # --------------------------
    # 1. 数据加载 (Data Loading)
    # --------------------------
    print(f"Loading data... (Source: {SOURCE_ROOT}, Target: {TARGET_ROOT})")

    # 训练集: 混合源域和目标域
    train_dataset = RoadUDADataset(
        source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
        source_list_name="train.txt", target_list_name="train.txt",  # 假设目标域列表也是 train.txt 或者是 A.txt，请确认文件名
        crop_size=CROP_SIZE, mode='train'
    )

    # 关键修改：drop_last=True 防止 batch size 不一致报错
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True
    )

    # 目标域验证集 (Target Val)
    target_val_dataset = RoadUDADataset(
        source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
        source_list_name="train.txt", target_list_name="train.txt",
        crop_size=CROP_SIZE, mode='val'
    )
    target_val_loader = DataLoader(target_val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 源域验证集 (Source Val)
    # 只有当你的 RoadUDADataset 支持 mode='val_source' 时才有效
    source_val_dataset = RoadUDADataset(
        source_root=SOURCE_ROOT, target_root=TARGET_ROOT,
        source_list_name="train.txt", target_list_name="train.txt",
        crop_size=CROP_SIZE, mode='val_source'
    )
    source_val_loader = DataLoader(source_val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # --------------------------
    # 2. 模型初始化 (Model Init)
    # --------------------------
    print("Initializing models...")
    # enc_invariant = ResNet34EncoderWithDIFA(pretrained=True).to(DEVICE)
    enc_invariant = DLinkNetEncoderWithDIFA(pretrained=True).to(DEVICE)
    enc_specific = ResNet34Encoder(pretrained=True).to(DEVICE)
    # decoder = DAN_Net_Decoder(num_classes=1).to(DEVICE)
    decoder = DLinkNetDecoder(num_classes=1).to(DEVICE)
    reconstruction_mod = ReconstructionModule(in_channels=512).to(DEVICE)

    # --------------------------
    # 3. 优化器 & 评估器
    # --------------------------
    params = itertools.chain(
        enc_invariant.parameters(), enc_specific.parameters(),
        decoder.parameters(), reconstruction_mod.parameters()
    )
    optimizer = optim.Adam(params, lr=LEARNING_RATE, betas=(0.9, 0.99))

    evaluator = Evaluator(enc_invariant, enc_specific, decoder, DEVICE)

    # 损失函数
    criterion_seg = SegmentationLoss()
    criterion_adv = AdversarialLoss()
    criterion_dis = DisentangleLoss()
    criterion_tem = TEMLoss()

    # 记录历史
    best_target_iou = 0.0
    # 正确的代码：初始化所有你需要用到的 key
    history = {
        'epoch': [],
        'train_loss': [],

        # 目标域指标
        'target_iou_inv': [],
        'target_iou_fused': [],
        'target_f1_inv': [],
        'target_f1_fused': [],

        # 源域指标
        'source_iou_inv': []
    }

    print(f"Start training on {DEVICE}...")

    for epoch in range(NUM_EPOCHS):
        # --- 训练阶段 (Training) ---
        enc_invariant.train()
        enc_specific.train()
        decoder.train()
        reconstruction_mod.train()

        epoch_loss = 0.0
        steps = len(train_loader)

        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch")

        for i, (img_s, mask_s, img_t) in enumerate(pbar):
            img_s, mask_s, img_t = img_s.to(DEVICE), mask_s.to(DEVICE), img_t.to(DEVICE)

            # GRL Alpha 动态调整 (0 -> 1)
            p = float(i + epoch * steps) / (NUM_EPOCHS * steps)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()
            # =================== Forward ===================
            # 1. 源域流 (Source Flow)
            feats_s, dom_preds_s = enc_invariant(img_s, alpha=alpha)
            pred_s = decoder(feats_s)
            loss_seg_s = criterion_seg(pred_s, mask_s)

            # --- [关键修改]：加入预热判断 ---
            WARMUP_EPOCHS = 10  # 设置 15 轮预热

            if epoch < WARMUP_EPOCHS:
                # 预热阶段：只训练源域分割
                total_loss = loss_seg_s

                # 为了日志好看，把其他损失（假的）也记录一下
                loss_adv_total = torch.tensor(0.0)
                loss_tem = torch.tensor(0.0)

            else:
                # 正常 UDA 训练阶段

                # 2. 目标域流 (Target Flow)
                loss_adv_s = criterion_adv(dom_preds_s, is_source=True)
                feats_inv_t, dom_preds_t = enc_invariant(img_t, alpha=alpha)
                loss_adv_t = criterion_adv(dom_preds_t, is_source=False)

                feats_spec_t = enc_specific(img_t)
                f4_inv, f4_spec = feats_inv_t[-1], feats_spec_t[-1]
                rec_img = reconstruction_mod(f4_inv, f4_spec, target_size=(CROP_SIZE, CROP_SIZE))
                loss_diff, loss_recon = criterion_dis(f4_inv, f4_spec, rec_img, img_t)

                # 3. 目标增强 (TEM)
                with torch.no_grad():
                    pred_inv_t = decoder(feats_inv_t)

                feats_fused_t = list(feats_inv_t)
                feats_fused_t[-1] = feats_inv_t[-1] + feats_spec_t[-1]
                pred_fused_t = decoder(feats_fused_t)
                loss_tem = criterion_tem(pred_fused_t, pred_inv_t)

                # =================== Backward ===================
                loss_adv_total = (loss_adv_s + loss_adv_t) * 0.5

                total_loss = (loss_seg_s +
                              LAMBDA_ADV * loss_adv_total +
                              LAMBDA_DIFF * loss_diff +
                              LAMBDA_RECON * loss_recon +
                              LAMBDA_TEM * loss_tem)

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.3f}",
                'Seg': f"{loss_seg_s.item():.3f}",
                # [修改]：确保 .item() 在 tensor 上调用
                'Adv': f"{loss_adv_total.item():.3f}",
                'TEM': f"{loss_tem.item():.3f}"
            })
            # # =================== Forward ===================
            # # 1. 源域流 (Source Flow)
            # feats_s, dom_preds_s = enc_invariant(img_s, alpha=alpha)
            # pred_s = decoder(feats_s)
            #
            # loss_seg_s = criterion_seg(pred_s, mask_s)
            # # loss_adv_s = criterion_adv(dom_preds_s, is_source=True)
            #
            # # 2. 目标域流 (Target Flow)
            # # 2.1 域不变路径
            # feats_inv_t, dom_preds_t = enc_invariant(img_t, alpha=alpha)
            # loss_adv_t = criterion_adv(dom_preds_t, is_source=False)
            #
            # # 2.2 域特定路径
            # feats_spec_t = enc_specific(img_t)
            #
            # # 2.3 解耦与重建
            # f4_inv, f4_spec = feats_inv_t[-1], feats_spec_t[-1]
            # rec_img = reconstruction_mod(f4_inv, f4_spec, target_size=(CROP_SIZE, CROP_SIZE))
            #
            # loss_diff, loss_recon = criterion_dis(f4_inv, f4_spec, rec_img, img_t)
            #
            # # 2.4 目标增强 (TEM)
            # with torch.no_grad():
            #     # Teacher: 仅用不变特征预测 (Detach, 不传梯度)
            #     pred_inv_t = decoder(feats_inv_t)
            #
            # # Student: 融合特征预测
            # feats_fused_t = list(feats_inv_t)
            # feats_fused_t[-1] = feats_inv_t[-1] + feats_spec_t[-1]
            # pred_fused_t = decoder(feats_fused_t)
            #
            # loss_tem = criterion_tem(pred_fused_t, pred_inv_t)
            #
            # # =================== Backward ===================
            # loss_adv_total = (loss_adv_s + loss_adv_t) * 0.5
            #
            # total_loss = (loss_seg_s +
            #               LAMBDA_ADV * loss_adv_total +
            #               LAMBDA_DIFF * loss_diff +
            #               LAMBDA_RECON * loss_recon +
            #               LAMBDA_TEM * loss_tem)
            #
            # total_loss.backward()
            # optimizer.step()
            #
            # epoch_loss += total_loss.item()
            #
            # # 更新进度条
            # pbar.set_postfix({
            #     'Loss': f"{total_loss.item():.3f}",
            #     'Seg': f"{loss_seg_s.item():.3f}",
            #     'Adv': f"{loss_adv_total.item():.3f}",
            #     'TEM': f"{loss_tem.item():.3f}"
            # })

        # --- End of Epoch ---
        avg_loss = epoch_loss / steps
        print(f"Epoch {epoch + 1} Finished. Avg Loss: {avg_loss:.4f}")

        # 保存最新的 Checkpoint (覆盖式)
        torch.save({
            'epoch': epoch,
            'model_state': enc_invariant.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, os.path.join(SAVE_DIR, 'latest_checkpoint.pth'))

        # --- 评估阶段 (Evaluation) ---
        # 策略：每 EVAL_INTERVAL 轮评估一次，或者是最后一轮
        # --- 评估阶段 (Evaluation) ---
        if (epoch + 1) % EVAL_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f"\n[Eval] Epoch {epoch + 1} Evaluation...")

            # 1. 评估目标域 (Target) -> 两个都看
            metrics_t = evaluator.evaluate(target_val_loader, desc="Eval Target")

            iou_t_inv = metrics_t['Inv']['IoU']
            iou_t_fused = metrics_t['Fused']['IoU']

            print(f" >> [Target Inv]   IoU: {iou_t_inv:.4f}, F1: {metrics_t['Inv']['F1']:.4f}")
            print(f" >> [Target Fused] IoU: {iou_t_fused:.4f}, F1: {metrics_t['Fused']['F1']:.4f} (Paper Key Metric)")

            # 2. 评估源域 (Source) -> 只看 Inv
            metrics_s = evaluator.evaluate(source_val_loader, desc="Eval Source")

            # 我们只关心源域的基础能力有没有崩，不关心源域融合特定特征（因为SpecificEncoder没学过源域）
            iou_s_inv = metrics_s['Inv']['IoU']
            print(f" >> [Source Inv]   IoU: {iou_s_inv:.4f} (Base Performance)")

            # 3. 记录日志
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_loss)

            # 记录详细信息
            history['target_iou_inv'].append(iou_t_inv)
            history['target_iou_fused'].append(iou_t_fused)
            history['source_iou_inv'].append(iou_s_inv)

            with open(os.path.join(SAVE_DIR, 'metrics.json'), 'w') as f:
                json.dump(history, f, indent=4)

            # 4. 保存最佳模型
            # 关键：使用【目标域融合特征】的 IoU 作为保存标准
            if iou_t_fused > best_target_iou:
                best_target_iou = iou_t_fused
                save_path = os.path.join(SAVE_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'enc_invariant': enc_invariant.state_dict(),
                    'enc_specific': enc_specific.state_dict(),  # 别忘了保存这个！融合预测需要它
                    'decoder': decoder.state_dict(),
                    'best_iou': best_target_iou
                }, save_path)
                print(f" !! New Best Model Saved (Fused IoU: {best_target_iou:.4f})")


# ==========================================
# 关键：必须放在 if __name__ == '__main__': 下
# 否则 Windows 下多进程会无限递归报错
# ==========================================
if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # --------------------------
    # 2. 只在主程序入口处创建文件夹
    # --------------------------
    TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = f'./checkpoints/yure_exp_{TIME_STR}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    train()