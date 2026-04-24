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
from model.cldice import soft_cldice_loss
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
WARMUP_EPOCHS = 0

# 损失函数权重
LAMBDA_ADV = 0.01  # 保留对抗权重
LAMBDA_FEAT_MIX = 0.01  # [新增] 区域解耦特征一致性损失权重 (建议 0.1)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = ""


def calc_prototype_loss1(feats_mix, feats_s, feats_t, mask_mix):
    """
    改进版：基于余弦相似度的原型对齐
    1. 使用 Cosine Similarity 代替 MSE，容忍背景带来的数值差异。
    2. 对目标特征 (feats_s, feats_t) 进行 detach，防止梯度传导错误。
    """
    loss = 0.0
    valid_layers = 0

    # 遍历多尺度特征
    for f_m, f_s, f_t in zip(feats_mix, feats_s, feats_t):
        N, C, H, W = f_m.size()

        # 下采样 Mask
        mask_resized = torch.nn.functional.interpolate(mask_mix, size=(H, W), mode='nearest')

        # ---------------------------
        # A. 前景原型 (Road)
        # ---------------------------
        # 计算 Mask=1 区域的特征总和
        feat_m_fg_sum = (f_m * mask_resized).sum(dim=(2, 3))
        feat_s_fg_sum = (f_s * mask_resized).sum(dim=(2, 3))

        mask_fg_count = mask_resized.sum(dim=(2, 3)) + 1e-6

        # 计算原型 (均值向量)
        proto_m_fg = feat_m_fg_sum / mask_fg_count  # (B, C)
        proto_s_fg = feat_s_fg_sum / mask_fg_count  # (B, C)

        # 【改进1】对源域原型 detach，把它当成固定的目标
        proto_s_fg = proto_s_fg.detach()

        # 【改进2】使用余弦相似度损失 (1 - cos_sim)
        # CosineEmbeddingLoss 需要输入 (x1, x2, target=1) 表示希望相似
        # 这里手动计算更直观: 1 - (A . B) / (|A| * |B|)
        proto_m_fg_norm = torch.nn.functional.normalize(proto_m_fg, p=2, dim=1)
        proto_s_fg_norm = torch.nn.functional.normalize(proto_s_fg, p=2, dim=1)

        # 两个向量越相似，点积越接近 1，Loss 越接近 0
        loss_fg = 1.0 - (proto_m_fg_norm * proto_s_fg_norm).sum(dim=1).mean()

        # ---------------------------
        # B. 背景原型 (Background)
        # ---------------------------
        mask_bg = 1 - mask_resized
        mask_bg_count = mask_bg.sum(dim=(2, 3)) + 1e-6

        loss_bg = 0.0
        if mask_bg_count.min() > 0:  # 确保有背景
            feat_m_bg_sum = (f_m * mask_bg).sum(dim=(2, 3))
            feat_t_bg_sum = (f_t * mask_bg).sum(dim=(2, 3))

            proto_m_bg = feat_m_bg_sum / mask_bg_count
            proto_t_bg = feat_t_bg_sum / mask_bg_count

            # 【改进1】Detach 目标域背景
            proto_t_bg = proto_t_bg.detach()

            # 【改进2】余弦相似度
            proto_m_bg_norm = torch.nn.functional.normalize(proto_m_bg, p=2, dim=1)
            proto_t_bg_norm = torch.nn.functional.normalize(proto_t_bg, p=2, dim=1)

            loss_bg = 1.0 - (proto_m_bg_norm * proto_t_bg_norm).sum(dim=1).mean()

        loss += (loss_fg + loss_bg)
        valid_layers += 1

    return loss / (valid_layers + 1e-6)
# --------------------------
def calc_prototype_loss(feats_mix, feats_s, feats_t, mask_mix):
    """
    原型对齐损失 (Prototype Alignment Loss)
    不对齐单个像素，而是对齐'道路类'和'背景类'的全局特征均值。
    """
    loss = 0.0
    for f_m, f_s, f_t in zip(feats_mix, feats_s, feats_t):
        N, C, H, W = f_m.size()

        # 1. 下采样 Mask
        mask_resized = torch.nn.functional.interpolate(mask_mix, size=(H, W), mode='nearest')

        # --- 前景原型对齐 (Road Prototype) ---
        # 计算混合图中，道路区域的特征均值 (B, C)
        # sum(dim=(2,3)) 是在空间 H,W 上求和
        feat_m_fg_sum = (f_m * mask_resized).sum(dim=(2, 3))
        mask_fg_sum = mask_resized.sum(dim=(2, 3)) + 1e-6  # 避免除0
        proto_m_fg = feat_m_fg_sum / mask_fg_sum  # (B, C)

        # 计算源域图中，道路区域的特征均值
        feat_s_fg_sum = (f_s * mask_resized).sum(dim=(2, 3))
        proto_s_fg = feat_s_fg_sum / mask_fg_sum  # (B, C)

        # 拉近两个原型的距离 (Cosine Similarity 或者 MSE)
        # 这里用 MSE 比较简单直接
        loss += torch.nn.functional.mse_loss(proto_m_fg, proto_s_fg)

        # --- 背景原型对齐 (Background Prototype) ---
        mask_bg = 1 - mask_resized
        feat_m_bg_sum = (f_m * mask_bg).sum(dim=(2, 3))
        mask_bg_sum = mask_bg.sum(dim=(2, 3)) + 1e-6
        proto_m_bg = feat_m_bg_sum / mask_bg_sum

        feat_t_bg_sum = (f_t * mask_bg).sum(dim=(2, 3))
        proto_t_bg = feat_t_bg_sum / mask_bg_sum

        loss += torch.nn.functional.mse_loss(proto_m_bg, proto_t_bg)

    return loss / len(feats_mix)

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

            # --- 1. 源域流 (Source Flow) ---
            # alpha 设置为 0.0，关闭 GRL (Gradient Reversal Layer) 的反向传播
            # 虽然 LAMBDA_ADV=0 已经够了，但把 alpha 设为 0 更保险
            feats_s, dom_preds_s = enc_invariant(img_s, alpha=0.0)
            pred_s = decoder(feats_s)
            loss_seg_s = criterion_seg(pred_s, mask_s)

            # --- 2. 目标域流 (Target Flow) ---
            feats_inv_t, dom_preds_t = enc_invariant(img_t, alpha=0.0)
            pred_t = decoder(feats_inv_t)

            # =============================================================
            # [修改开始] 消融实验：禁用对抗损失 (No Adversarial Loss)
            # =============================================================

            # 1. 强制将对抗损失权重设为 0
            loss_adv_total = torch.tensor(0.0).to(DEVICE)
            LAMBDA_ADV = 0.0

            # 2. 混合一致性模块 (保持开启，因为你只想消融掉对抗)
            #    注意：Warmup 逻辑可能需要调整，因为没有对抗了，可以直接开启混合

            # 这里原本有 if epoch < WARMUP_EPOCHS 的判断
            # 但既然去掉了对抗，我们可以让混合模块从第 0 轮或者是 WARMUP 后直接开始
            if epoch < WARMUP_EPOCHS:
                # Warmup 期间只训练源域分割
                loss_mix_consistency = torch.tensor(0.0).to(DEVICE)
                loss_feat_mix = torch.tensor(0.0).to(DEVICE)
                LAMBDA_MIX = 0.0
                LAMBDA_FEAT_MIX = 0.0
            else:
                # --- [保持开启] 中间域混合模块 ---

                # A. 准备蒙版
                if mask_s.dim() == 3:
                    mask_mix = mask_s.unsqueeze(1).float()
                else:
                    mask_mix = mask_s.float()

                # B. 制作混合图像
                img_mix = img_s * mask_mix + img_t * (1 - mask_mix)

                # C. 混合图像前向传播
                feats_mix, _ = enc_invariant(img_mix, alpha=0.0)
                pred_mix = decoder(feats_mix)

                # D. 计算特征损失 (保持)
                loss_feat_mix = calc_prototype_loss1(
                    feats_mix, feats_s, feats_inv_t, mask_mix
                )

                # E. 设置混合权重 (保持原逻辑)
                if epoch < 50:
                    HIGH_THRESH = 0.6;
                    LOW_THRESH = 0.3;
                    LAMBDA_MIX = 1;
                    LAMBDA_FEAT_MIX = 0.1
                elif epoch < 100:
                    HIGH_THRESH = 0.7;
                    LOW_THRESH = 0.2;
                    LAMBDA_MIX = 1;
                    LAMBDA_FEAT_MIX = 0.1
                else:
                    HIGH_THRESH = 0.8;
                    LOW_THRESH = 0.1;
                    LAMBDA_MIX = 1;
                    LAMBDA_FEAT_MIX = 0.1

                # F. 构建伪标签 & 计算混合一致性损失 (保持)
                with torch.no_grad():
                    prob_t = torch.sigmoid(pred_t.detach())
                    pseudo_label_t = (prob_t > HIGH_THRESH).float()
                    conf_mask_t = (prob_t > HIGH_THRESH) | (prob_t < LOW_THRESH)
                    conf_mask_t = conf_mask_t.float()
                    target_mix = mask_mix * 1.0 + pseudo_label_t * (1 - mask_mix)
                    pixel_weight = conf_mask_t * (1 - mask_mix)

                bce_criterion = nn.BCEWithLogitsLoss(reduction='none')
                loss_pixel_map = bce_criterion(pred_mix, target_mix)
                loss_weighted = loss_pixel_map * pixel_weight
                loss_mix_consistency = loss_weighted.sum() / (pixel_weight.sum() + 1e-6)

                # --- [已注释] 对抗损失计算 ---
                # 原本这里会计算 criterion_adv，现在直接注释掉
                """
                loss_adv_s = criterion_adv(dom_preds_s, is_source=True)
                loss_adv_t = criterion_adv(dom_preds_t, is_source=False)
                loss_adv_total = (loss_adv_s + loss_adv_t) * 0.5
                """

            # =============================================================
            # [修改结束]
            # =============================================================

            # --- 5. 总损失 ---
            # 此时 LAMBDA_ADV 为 0，loss_adv_total 为 0
            total_loss = loss_seg_s + \
                         LAMBDA_ADV * loss_adv_total + \
                         LAMBDA_MIX * loss_mix_consistency + \
                         LAMBDA_FEAT_MIX * loss_feat_mix

            # --- 反向传播 ---
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.3f}",
                'Seg': f"{loss_seg_s.item():.3f}",
                'Adv': f"{loss_adv_total.item():.3f}",
                'Mix': f"{loss_mix_consistency.item():.3f}",
                'Feat': f"{loss_feat_mix.item():.4f}"  # [新增] 显示特征损失
            })

            import torchvision.utils as vutils

            # if i % 10 == 0:
            #     # 还原一下看起来比较直观
            #     # 1. 混合后的原始图片
            #     vutils.save_image(pred_mix, f"{SAVE_DIR}/vis_img_mix_{epoch}_{i}.png", normalize=True)
            #
            #     # 2. 最终生成的 Target (注意：这里全是0和1)
            #     vutils.save_image(target_mix, f"{SAVE_DIR}/vis_target_{epoch}_{i}.png")
            #
            #     # 3. 权重图 (可以看到哪里被忽略了，黑色的就是被忽略的区域)
            #     vutils.save_image(pixel_weight, f"{SAVE_DIR}/vis_weight_{epoch}_{i}.png")

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
    SAVE_DIR = f'./checkpoints/ablation_mix_zhong_noADV{TIME_STR}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    train()