import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt
from DADataset_CHN6 import RoadUDADataset
# ================= 配置区域 =================
# 1. 路径配置
SOURCE_ROOT = r"D:\JinWenBo\CHN6-CUG"
TARGET_ROOT = r"C:\Users\Administrator\Desktop\RoadData\deepglobe"

# 2. 文件列表名称
# 请确保 D:\JinWenBo\CHN6-CUG\source_domain_list.txt 存在
SOURCE_LIST_NAME = "source_domain_list.txt"
TARGET_LIST_NAME = "train.txt"

# 3. 参数配置
CROP_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = 2  # Windows下建议不要设置太高，2或4即可


# ================= Dataset 定义 =================
# 为了保证这一个脚本能独立运行，我把 RoadUDADataset 类直接写在这里



# ================= 工具函数：反标准化 =================
def denormalize(tensor):
    """ (C,H,W) Tensor -> (H,W,C) Numpy, 范围 [0,1] """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img



if __name__ == '__main__':

    print("🚀 正在初始化 Dataset...")

    # 1. 实例化 Dataset
    train_dataset = RoadUDADataset(
        source_root=SOURCE_ROOT,
        target_root=TARGET_ROOT,
        source_list_name=SOURCE_LIST_NAME,
        target_list_name=TARGET_LIST_NAME,
        crop_size=CROP_SIZE,
        mode='train'
    )

    print(f"   Dataset 初始化完成。源域样本数: {len(train_dataset)}")

    # 2. 实例化 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    print("🔍 开始读取一个 Batch 进行可视化验证...")

    try:
        # 3. 获取数据
        batch = next(iter(train_loader))
        src_imgs, src_masks, tgt_imgs = batch

        print(f"✅ 数据读取成功!")
        print(f"   Source Image Shape: {src_imgs.shape}")
        print(f"   Source Mask Shape : {src_masks.shape}")
        print(f"   Target Image Shape: {tgt_imgs.shape}")

        # 4. 可视化
        plt.figure(figsize=(12, 8))
        plt.suptitle("Dataloader Verification: Source vs Target", fontsize=16)

        num_show = min(BATCH_SIZE, 4)
        for i in range(num_show):
            # 反标准化
            s_img = denormalize(src_imgs[i])
            s_mask = src_masks[i].squeeze().cpu().numpy()
            t_img = denormalize(tgt_imgs[i])

            # 显示源域图
            plt.subplot(num_show, 3, i * 3 + 1)
            plt.imshow(s_img)
            plt.title(f"Source Img {i}")
            plt.axis('off')

            # 显示源域Mask
            plt.subplot(num_show, 3, i * 3 + 2)
            plt.imshow(s_mask, cmap='gray')
            plt.title(f"Source Mask {i}")
            plt.axis('off')

            # 显示目标域图
            plt.subplot(num_show, 3, i * 3 + 3)
            plt.imshow(t_img)
            plt.title(f"Target Img {i}")
            plt.axis('off')

        plt.tight_layout()
        print("🖼️  窗口已弹出，请检查图片内容。")
        plt.show()

    except Exception as e:
        print("\n❌ 发生错误:")
        print(e)
        import traceback

        traceback.print_exc()

    # # 验证结束提示
    # input("按回车键退出程序...")