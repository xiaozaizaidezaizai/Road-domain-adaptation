import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms


class RoadUDADataset(Dataset):
    def __init__(self, source_root, target_root,
                 source_list_name="source_domain_list.txt",  # 注意这里默认值改了
                 target_list_name="train.txt",
                 crop_size=512, mode='train'):

        self.source_root = source_root
        self.target_root = target_root
        self.crop_size = crop_size
        self.mode = mode

        # --- 1. 解析源域 (CHN6-CUG) [修改处] ---
        self.source_items = []
        source_list_path = os.path.join(source_root, source_list_name)

        if os.path.exists(source_list_path):
            with open(source_list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue

                    # 之前的 line 是 "train/am100003"
                    # CHN6-CUG 的命名规则是:
                    # 图片: train/am100003_sat.jpg
                    # 标签: train/am100003_mask.png

                    img_path = os.path.join(source_root, f"{line}_sat.jpg")
                    mask_path = os.path.join(source_root, f"{line}_mask.png")

                    # (可选) 可以在这里加一个 check，防止文件不存在报错
                    # if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.source_items.append((img_path, mask_path))
        else:
            print(f"[警告] 源域列表文件不存在: {source_list_path}")

        # --- 2. 解析目标域 (DeepGlobe) [保持不变] ---
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

        # 打印一下加载数量，确保路径对不对
        if mode == 'train':
            print(f"[Dataset] 加载源域 (CHN6-CUG): {len(self.source_items)} 张")
            print(f"[Dataset] 加载目标域 (DeepGlobe): {len(self.target_items)} 张")

    # ... _transform, __getitem__, __len__ 保持你原来的代码不变 ...
    # (这里省略不写，直接用你原来提供的即可)
    # def _transform(self, image, mask):
    #     # ... (保持原样) ...
    #     # 注意: 即使 CHN6-CUG 图片已经是 512，Resize(1024)再Crop(512) 也是一种有效的数据增强
    #     if self.mode == 'train':
    #         image = TF.resize(image, (1024, 1024), interpolation=Image.BILINEAR)
    #         if mask:
    #             mask = TF.resize(mask, (1024, 1024), interpolation=Image.NEAREST)
    #         i, j, h, w = transforms.RandomCrop.get_params(
    #             image, output_size=(self.crop_size, self.crop_size))
    #         image = TF.crop(image, i, j, h, w)
    #         if mask:
    #             mask = TF.crop(mask, i, j, h, w)
    #         if random.random() > 0.5:
    #             image = TF.hflip(image)
    #             if mask: mask = TF.hflip(mask)
    #         if random.random() > 0.5:
    #             image = TF.vflip(image)
    #             if mask: mask = TF.vflip(mask)
    #     else:
    #         image = TF.resize(image, (1024, 1024), interpolation=Image.BILINEAR)
    #         if mask:
    #             mask = TF.resize(mask, (1024, 1024), interpolation=Image.NEAREST)
    #         image = TF.center_crop(image, (self.crop_size, self.crop_size))
    #         if mask:
    #             mask = TF.center_crop(mask, (self.crop_size, self.crop_size))
    #
    #     image = TF.to_tensor(image)
    #     image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     if mask:
    #         mask = TF.to_tensor(mask)
    #         mask = (mask > 0.5).float()
    #     return image, mask

    def _transform(self, image, mask):
        # 获取当前图片的宽和高
        w, h = image.size

        # ==========================================
        # 训练模式 (Train)
        # ==========================================
        if self.mode == 'train':
            # 1. 安全检查：只有当图片小于 512 时才放大 (防止报错)
            #    CHN6(512) 和 DeepGlobe(1024) 都会跳过这一步，保持原分辨率
            if w < self.crop_size or h < self.crop_size:
                image = TF.resize(image, (self.crop_size, self.crop_size), interpolation=Image.BILINEAR)
                if mask:
                    mask = TF.resize(mask, (self.crop_size, self.crop_size), interpolation=Image.NEAREST)

            # 2. 随机裁剪 (Random Crop)
            #    - 如果是 CHN6 (512x512): 这里实际上就是取全图，不进行裁剪
            #    - 如果是 DeepGlobe (1024x1024): 这里会随机切出一个 512x512 的局部
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.crop_size, self.crop_size))

            image = TF.crop(image, i, j, h, w)
            if mask:
                mask = TF.crop(mask, i, j, h, w)

            # 3. 随机翻转 (增强数据多样性)
            if random.random() > 0.5:
                image = TF.hflip(image)
                if mask: mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                if mask: mask = TF.vflip(mask)

        # ==========================================
        # 验证模式 (Val)
        # ==========================================
        else:
            # 验证集逻辑：如果是大图就取中心，如果是 512 就直接用
            if w < self.crop_size or h < self.crop_size:
                image = TF.resize(image, (self.crop_size, self.crop_size), interpolation=Image.BILINEAR)
                if mask:
                    mask = TF.resize(mask, (self.crop_size, self.crop_size), interpolation=Image.NEAREST)

            # Center Crop
            # CHN6 (512) -> 取中心 512 (即全图)
            # DeepGlobe (1024) -> 取中心 512 区域
            image = TF.center_crop(image, (self.crop_size, self.crop_size))
            if mask:
                mask = TF.center_crop(mask, (self.crop_size, self.crop_size))

        # ==========================================
        # 标准化 (ToTensor & Normalize)
        # ==========================================
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if mask:
            mask = TF.to_tensor(mask)
            mask = (mask > 0.5).float()  # 二值化

        return image, mask


    def __getitem__(self, index):
        # ... (保持原样) ...
        if self.mode == 'train':
            src_idx = index
            tgt_idx = random.randint(0, len(self.target_items) - 1)
            src_path, src_mask_path = self.source_items[src_idx]
            tgt_path, _ = self.target_items[tgt_idx]
            src_img = Image.open(src_path).convert('RGB')
            src_mask = Image.open(src_mask_path).convert('L')
            tgt_img = Image.open(tgt_path).convert('RGB')
            src_img, src_mask = self._transform(src_img, src_mask)
            tgt_img, _ = self._transform(tgt_img, None)
            return src_img, src_mask, tgt_img
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
        elif self.mode == 'val_source':
            src_idx = index % len(self.source_items)
            src_path, src_mask_path = self.source_items[src_idx]
            src_img = Image.open(src_path).convert('RGB')
            src_mask = Image.open(src_mask_path).convert('L')
            src_img, src_mask = self._transform(src_img, src_mask)
            return src_img, src_mask, os.path.basename(src_path)

    def __len__(self):
        # ... (保持原样) ...
        if self.mode == 'train':
            return len(self.source_items)
        elif self.mode == 'val':
            return len(self.target_items)
        elif self.mode == 'val_source':
            return len(self.source_items)