import torch
import torch.nn as nn

# 导入你的自定义模块 (保持和你的项目路径一致)
from model.DANnet import ResNet34Encoder
from model.DlinkNet_Encoder import DLinkNetEncoderWithDIFA, DLinkNetDecoder
from model.Reconstruction import ReconstructionModule

try:
    from thop import profile, clever_format
except ImportError:
    print("[错误] 未找到 thop 库，请先在终端运行: pip install thop")
    exit()

# --------------------------
# 配置参数
# --------------------------
CROP_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------
# 定义包装类 (用于串联计算图)
# --------------------------
class FullModelWrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # 模拟前向传播，alpha 给定默认值
        feats, _ = self.encoder(x, alpha=0.0)
        out = self.decoder(feats)
        return out


def main():
    print(f"正在使用设备: {DEVICE}")
    print("正在初始化模型...\n")

    # 1. 初始化模型
    enc_invariant = DLinkNetEncoderWithDIFA(pretrained=False).to(DEVICE)
    enc_specific = ResNet34Encoder(pretrained=False).to(DEVICE)
    decoder = DLinkNetDecoder(num_classes=1).to(DEVICE)
    reconstruction_mod = ReconstructionModule(in_channels=512).to(DEVICE)

    # 2. 计算可训练参数量 (Parameters)
    # 分别统计各个模块
    params_inv = sum(p.numel() for p in enc_invariant.parameters())
    params_spec = sum(p.numel() for p in enc_specific.parameters())
    params_dec = sum(p.numel() for p in decoder.parameters())
    params_rec = sum(p.numel() for p in reconstruction_mod.parameters())

    # 你在消融实验中实际更新的参数 (Invariant Encoder + Decoder)
    active_params = params_inv + params_dec
    total_params = params_inv + params_spec + params_dec + params_rec

    print("-" * 40)
    print("【参数量统计 (Parameters)】")
    print(f" - DLinkNetEncoderWithDIFA: {params_inv / 1e6:.2f} M")
    print(f" - ResNet34Encoder (特定域): {params_spec / 1e6:.2f} M")
    print(f" - DLinkNetDecoder:         {params_dec / 1e6:.2f} M")
    print(f" - ReconstructionModule:    {params_rec / 1e6:.2f} M")
    print(f" => 实际训练(活跃)参数量:   {active_params / 1e6:.2f} M")
    print(f" => 模型总参数量:           {total_params / 1e6:.2f} M")
    print("-" * 40)

    # 3. 计算计算量 (FLOPs / MACs)
    # 在推理时，通常只走 Invariant Encoder -> Decoder 这条路
    print("\n正在计算 FLOPs (输入尺寸: 1 x 3 x {} x {})...".format(CROP_SIZE, CROP_SIZE))

    eval_model = FullModelWrapper(enc_invariant, decoder).to(DEVICE)
    eval_model.eval()

    # 构造标准单图输入
    dummy_input = torch.randn(1, 3, CROP_SIZE, CROP_SIZE).to(DEVICE)

    # 使用 thop 进行前向传播统计
    macs, params = profile(eval_model, inputs=(dummy_input,), verbose=False)

    # 格式化输出为 G, M 等单位
    macs_str, params_str = clever_format([macs, params], "%.2f")

    print("-" * 40)
    print("【计算量统计 (MACs/FLOPs)】")
    print(f" - 测试管线: Invariant Encoder -> Decoder")
    print(f" - 乘加操作数 (MACs): {macs_str}")
    print(f" - 对应参数量 (Params): {params_str}")
    print("-" * 40)
    print("提示: 1 MAC 约等于 2 FLOPs，在论文中通常直接将 MACs 作为 FLOPs 报告。")


if __name__ == '__main__':
    # 避免 OMP 报错
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    with torch.no_grad():
        main()