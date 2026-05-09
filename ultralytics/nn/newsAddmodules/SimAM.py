import torch
import torch.nn as nn

# ✅ 正确的 YOLO 适配版
class SimAM(nn.Module):
    def __init__(self, c1=None, c2=None, e_lambda=1e-4):  # 加入 c1, c2 占位
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)