import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms


class RoadUDADataset(Dataset):
    def __init__(self, source_root, target_root,
                 source_list_name="train.txt", target_list_name="train_cleaned.txt",
                 crop_size=512, mode='train', is_mass_target=True):

        self.source_root = source_root
        self.target_root = target_root
        self.crop_size = crop_size
        self.mode = mode
        self.is_mass_target = is_mass_target  # 标记是否使用 Massachusetts 策略

        # --- 1. 解析源域 (SpaceNet) ---
        # (保持你原有的源域逻辑不变)
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

        # --- 2. 解析目标域 (Massachusetts) ---
        self.target_items = []
        target_list_path = os.path.join(target_root, target_list_name)

        # 假设 Massachusetts 的文件夹结构是 tiff/train 和 tiff/train_labels
        # 根据你的截图
        tgt_img_dir = os.path.join(target_root, 'tiff', 'train')
        tgt_mask_dir = os.path.join(target_root, 'tiff', 'train_labels')

        if os.path.exists(target_list_path):
            with open(target_list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue

                    # [关键修改] Massachusetts 文件名需要加上后缀 _15.tif
                    # txt里是 "10078675"，实际文件是 "10078675_15.tif"
                    img_name = f"{line}_15.tiff"
                    mask_name = f"{line}_15.tif"  # 标签名通常和图片名一致

                    img_path = os.path.join(tgt_img_dir, img_name)
                    mask_path = os.path.join(tgt_mask_dir, mask_name)

                    self.target_items.append((img_path, mask_path))

        print(f"Loaded {len(self.source_items)} source images.")
        print(f"Loaded {len(self.target_items)} original target images (Mass).")
        if self.is_mass_target:
            print(f" -> Will be expanded to {len(self.target_items) * 4} patches (750->512).")

    # def _get_mass_patch(self, index):
    #     """
    #     专门处理 Massachusetts 数据集的切分逻辑
    #     逻辑：1张大图 -> 4张小图
    #     """
    #     # 1. 找到原始大图的索引
    #     original_idx = index // 4  # 整除，找到是哪张大图
    #     crop_pos = index % 4  # 取余，找到切哪个角
    #
    #     img_path, mask_path = self.target_items[original_idx]
    #
    #     image = Image.open(img_path).convert('RGB')
    #     # 只有在验证时才需要读 Label，训练时的无监督部分其实不需要，但读进来也没错
    #     try:
    #         mask = Image.open(mask_path).convert('L')
    #     except:
    #         mask = Image.new('L', image.size, 0)  # 如果没有标签，给全黑
    #
    #     # 2. 切分坐标 (Left, Top, Right, Bottom)
    #     # 原图 1500x1500 -> 切成 750x750
    #     w, h = 1500, 1500
    #     mid_w, mid_h = 750, 750
    #
    #     if crop_pos == 0:  # 左上
    #         box = (0, 0, mid_w, mid_h)
    #     elif crop_pos == 1:  # 右上
    #         box = (mid_w, 0, w, mid_h)
    #     elif crop_pos == 2:  # 左下
    #         box = (0, mid_h, mid_w, h)
    #     else:  # 右下
    #         box = (mid_w, mid_h, w, h)
    #
    #     image = image.crop(box)
    #     mask = mask.crop(box)
    #
    #     # 3. 缩放到 512x512
    #     image = TF.resize(image, (self.crop_size, self.crop_size), interpolation=Image.BILINEAR)
    #     mask = TF.resize(mask, (self.crop_size, self.crop_size), interpolation=Image.NEAREST)
    #
    #     return image, mask, os.path.basename(img_path)
    def _get_mass_patch(self, index):
        """
        逻辑修改：1张大图 -> 4张小图(750x750) -> 中心裁剪为 512x512
        """
        # 1. 找到原始大图的索引
        original_idx = index // 4
        crop_pos = index % 4

        img_path, mask_path = self.target_items[original_idx]

        image = Image.open(img_path).convert('RGB')
        try:
            mask = Image.open(mask_path).convert('L')
        except:
            mask = Image.new('L', image.size, 0)

        # 2. 切分坐标 (Left, Top, Right, Bottom)
        # 原图 1500x1500 -> 切成 750x750 的四个角
        w, h = 1500, 1500
        mid_w, mid_h = 750, 750

        if crop_pos == 0:  # 左上
            box = (0, 0, mid_w, mid_h)
        elif crop_pos == 1:  # 右上
            box = (mid_w, 0, w, mid_h)
        elif crop_pos == 2:  # 左下
            box = (0, mid_h, mid_w, h)
        else:  # 右下
            box = (mid_w, mid_h, w, h)

        # 此时 image 和 mask 都是 750x750
        image = image.crop(box)
        mask = mask.crop(box)

        # -----------------------------------------------------------
        # [修改点] 不缩放，改为中心裁剪
        # -----------------------------------------------------------
        # 以前是 resize (会变形/模糊)
        # 现在是 center_crop (保留原分辨率，只取中间 512x512)

        # 确保你的 self.crop_size 是 512
        image = TF.center_crop(image, (self.crop_size, self.crop_size))
        mask = TF.center_crop(mask, (self.crop_size, self.crop_size))

        return image, mask, os.path.basename(img_path)

    def _transform(self, image, mask):
        # 普通的增强逻辑 (用于源域，或者非Mass数据的目标域)
        if self.mode == 'train':
            image = TF.resize(image, (1024, 1024), interpolation=Image.BILINEAR)
            if mask: mask = TF.resize(mask, (1024, 1024), interpolation=Image.NEAREST)

            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            image = TF.crop(image, i, j, h, w)
            if mask: mask = TF.crop(mask, i, j, h, w)

            if random.random() > 0.5:
                image = TF.hflip(image)
                if mask: mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                if mask: mask = TF.vflip(mask)
        else:
            image = TF.resize(image, (1024, 1024), interpolation=Image.BILINEAR)
            if mask: mask = TF.resize(mask, (1024, 1024), interpolation=Image.NEAREST)
            image = TF.center_crop(image, (self.crop_size, self.crop_size))
            if mask: mask = TF.center_crop(mask, (self.crop_size, self.crop_size))

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if mask:
            mask = TF.to_tensor(mask)
            mask = (mask > 0.5).float()
        return image, mask

    def __getitem__(self, index):
        # --- 模式 A: 训练 ---
        if self.mode == 'train':
            # 源域逻辑不变
            src_idx = index % len(self.source_items)
            src_path, src_mask_path = self.source_items[src_idx]
            src_img = Image.open(src_path).convert('RGB')
            src_mask = Image.open(src_mask_path).convert('L')

            # 源域做普通的 Transform
            src_img, src_mask = self._transform(src_img, src_mask)

            # --- 目标域逻辑 (Massachusetts 特供版) ---
            if self.is_mass_target:
                # 随机选一张图的一个角 (Total Items = len * 4)
                tgt_total_len = len(self.target_items) * 4
                tgt_rand_idx = random.randint(0, tgt_total_len - 1)

                tgt_img, _, _ = self._get_mass_patch(tgt_rand_idx)

                # Mass 数据不需要再做 Crop 了，因为 _get_mass_patch 已经Resize到了512
                # 但仍然需要做 ToTensor 和 Normalize
                tgt_img = TF.to_tensor(tgt_img)
                tgt_img = TF.normalize(tgt_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            else:
                # 旧的目标域逻辑 (DeepGlobe)
                tgt_idx = random.randint(0, len(self.target_items) - 1)
                tgt_path, _ = self.target_items[tgt_idx]
                tgt_img = Image.open(tgt_path).convert('RGB')
                tgt_img, _ = self._transform(tgt_img, None)

            return src_img, src_mask, tgt_img

        # --- 模式 B: 验证目标域 (Massachusetts) ---
        elif self.mode == 'val':
            if self.is_mass_target:
                # 获取切片后的图
                tgt_img, tgt_mask, name = self._get_mass_patch(index)

                # ToTensor
                tgt_img = TF.to_tensor(tgt_img)
                tgt_img = TF.normalize(tgt_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                tgt_mask = TF.to_tensor(tgt_mask)
                tgt_mask = (tgt_mask > 0.5).float()

                return tgt_img, tgt_mask, name
            else:
                # 旧逻辑
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
        if self.mode == 'train':
            return len(self.source_items)
        elif self.mode == 'val':
            if self.is_mass_target:
                # 目标域长度变成了原来的 4 倍
                return len(self.target_items) * 4
            else:
                return len(self.target_items)
        elif self.mode == 'val_source':
            return len(self.source_items)