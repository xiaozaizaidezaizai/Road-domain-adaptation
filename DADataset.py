import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms

class RoadUDADataset(Dataset):
    def __init__(self, source_root, target_root,
                 source_list_name="train.txt", target_list_name="A.txt",
                 crop_size=512, mode='train'):

        self.source_root = source_root
        self.target_root = target_root
        self.crop_size = crop_size
        self.mode = mode

        # --- 1. 解析源域 (SpaceNet) ---
        self.source_items = []
        source_list_path = os.path.join(source_root, source_list_name)
        if os.path.exists(source_list_path):
            with open(source_list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    img_path = os.path.join(source_root, line)
                    mask_rel_path = line.replace('sats', 'maps')
                    mask_path = os.path.join(source_root, mask_rel_path)
                    self.source_items.append((img_path, mask_path))

        # --- 2. 解析目标域 (DeepGlobe) ---
        self.target_items = []
        target_list_path = os.path.join(target_root, target_list_name)
        tgt_img_dir = os.path.join(target_root, 'train', 'sat')
        tgt_mask_dir = os.path.join(target_root, 'train', 'mask')

        if os.path.exists(target_list_path):
            with open(target_list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    img_path = os.path.join(tgt_img_dir, f"{line}_sat.jpg")
                    mask_path = os.path.join(tgt_mask_dir, f"{line}_mask.png")
                    self.target_items.append((img_path, mask_path))

    def _transform(self, image, mask):

        # ---------------------
        # 训练模式：随机裁剪
        # ---------------------
        if self.mode == 'train':
            # 统一 Resize 到 1024x1024 (保证尺度一致)
            # 这一步在你的优化版里省略了，但为了评估逻辑统一，我们先加回来
            # (如果速度慢，再用优化版逻辑)
            image = TF.resize(image, (1024, 1024), interpolation=Image.BILINEAR)
            if mask:
                mask = TF.resize(mask, (1024, 1024), interpolation=Image.NEAREST)

            # 在 1024x1024 上随机切 512x512
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.crop_size, self.crop_size))
            image = TF.crop(image, i, j, h, w)
            if mask:
                mask = TF.crop(mask, i, j, h, w)

            # 随机翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                if mask: mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                if mask: mask = TF.vflip(mask)

        # ---------------------
        # [关键修复] 验证模式：中心裁剪
        # ---------------------
        else:  # 'val' or 'val_source'
            # 1. 先 Resize 到 1024x1024 (保持和训练时一样的基础尺度)
            image = TF.resize(image, (1024, 1024), interpolation=Image.BILINEAR)
            if mask:
                mask = TF.resize(mask, (1024, 1024), interpolation=Image.NEAREST)

            # 2. 取中心 512x512
            # 这样保证了输入模型的尺度永远是 512x512
            image = TF.center_crop(image, (self.crop_size, self.crop_size))
            if mask:
                mask = TF.center_crop(mask, (self.crop_size, self.crop_size))

        # ToTensor & Normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if mask:
            mask = TF.to_tensor(mask)
            mask = (mask > 0.5).float()

        return image, mask

    def __getitem__(self, index):
        # --- 模式 A: 训练 ---
        if self.mode == 'train':
            # (使用修改后的 __len__，以源域为准)
            src_idx = index
            tgt_idx = random.randint(0, len(self.target_items) - 1)

            src_path, src_mask_path = self.source_items[src_idx]
            tgt_path, _ = self.target_items[tgt_idx]

            src_img = Image.open(src_path).convert('RGB')
            src_mask = Image.open(src_mask_path).convert('L')
            tgt_img = Image.open(tgt_path).convert('RGB')

            # 训练时 src 和 tgt 都用 _transform
            src_img, src_mask = self._transform(src_img, src_mask)
            tgt_img, _ = self._transform(tgt_img, None)

            return src_img, src_mask, tgt_img

        # --- 模式 B: 验证目标域 ---
        elif self.mode == 'val':
            tgt_idx = index % len(self.target_items)
            tgt_path, tgt_mask_path = self.target_items[tgt_idx]

            tgt_img = Image.open(tgt_path).convert('RGB')
            try:
                tgt_mask = Image.open(tgt_mask_path).convert('L')
            except:
                tgt_mask = Image.new('L', tgt_img.size, 0)

            tgt_img, tgt_mask = self._transform(tgt_img, tgt_mask)
            return tgt_img, tgt_mask, os.path.basename(tgt_path)

        # --- 模式 C: 验证源域 ---
        elif self.mode == 'val_source':
            src_idx = index % len(self.source_items)
            src_path, src_mask_path = self.source_items[src_idx]
            src_img = Image.open(src_path).convert('RGB')
            src_mask = Image.open(src_mask_path).convert('L')
            src_img, src_mask = self._transform(src_img, src_mask)
            return src_img, src_mask, os.path.basename(src_path)

    def __len__(self):
        # [关键修复] 以源域为 Epoch 基准
        if self.mode == 'train':
            return len(self.source_items)
        elif self.mode == 'val':
            return len(self.target_items)
        elif self.mode == 'val_source':
            return len(self.source_items)