import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter=10):
        """
        用于提取 Soft Skeleton 的模块。
        Args:
            num_iter: 迭代次数。取决于你的道路宽度。
                      如果路很宽，需要更大的 iter 才能腐蚀到中心线。
                      对于一般的遥感道路，5-10 也就是够了。
        """
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        # 使用 max_pool 实现腐蚀 (Min pooling = - Max pooling of -x)
        p1 = -F.max_pool2d(-img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        return p1

    def soft_dilate(self, img):
        # 使用 max_pool 实现膨胀
        return F.max_pool2d(img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def soft_open(self, img):
        # 开运算 = 先腐蚀后膨胀
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        # 迭代腐蚀并提取骨架
        skel = F.relu(img - self.soft_open(img))

        for i in range(self.num_iter):
            img = self.soft_erode(img)
            # 骨架 = 当前图像 - 开运算后的图像
            delta = F.relu(img - self.soft_open(img))
            # 累加骨架 (类似逻辑或)
            skel = skel + F.relu(delta - skel * delta)

        return skel


class soft_cldice_loss(nn.Module):
    def __init__(self, iter_=5, smooth=1.0):
        super(soft_cldice_loss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: 预测图 (B, C, H, W)，建议先过 Sigmoid 变为概率 [0,1]
            y_true: 标签图 (B, C, H, W)，0 或 1
        """
        # 1. 提取骨架
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true.float())

        # 2. 计算拓扑精度 (T_prec): 预测的骨架有多少在真实掩码内
        # 这里的相乘相当于“交集”
        t_prec = (torch.sum(skel_pred * y_true) + self.smooth) / \
                 (torch.sum(skel_pred) + self.smooth)

        # 3. 计算拓扑召回 (T_sens): 真实的骨架有多少在预测掩码内
        t_sens = (torch.sum(skel_true * y_pred) + self.smooth) / \
                 (torch.sum(skel_true) + self.smooth)

        # 4. 计算 clDice
        cl_dice = 2.0 * (t_prec * t_sens) / (t_prec + t_sens)

        return 1.0 - cl_dice