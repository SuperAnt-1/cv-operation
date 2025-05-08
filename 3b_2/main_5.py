#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

# ---------------------------
# 1. 定义模型
# ---------------------------

class cGenerator(nn.Module):
    def __init__(self, nz, n_classes, ngf, nc):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        input_dim = nz + n_classes
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf,   nc,    4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # noise: (B, nz, 1, 1); labels: (B,)
        label_input = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        gen_input = torch.cat([noise, label_input], dim=1)
        return self.main(gen_input)


class cDiscriminator(nn.Module):
    def __init__(self, nc, ndf, n_classes):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.main = nn.Sequential(
            nn.Conv2d(nc + n_classes, ndf,   4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf,            ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),    nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2,          ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),    nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4,          ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),    nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8,          1,     4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # img: (B, nc, 64, 64); labels: (B,)
        label_input = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_map   = label_input.repeat(1, 1, img.size(2), img.size(3))
        d_input     = torch.cat([img, label_map], dim=1)
        return self.main(d_input).view(-1)


# ---------------------------
# 2. 主程序：生成并保存
# ---------------------------

if __name__ == "__main__":
    # 超参数
    nz, n_classes = 100, 10
    ngf, ndf, nc  = 64, 64, 3
    batch_size    = 16

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型实例化
    netG = cGenerator(nz, n_classes, ngf, nc).to(device)
    netD = cDiscriminator(nc, ndf, n_classes).to(device)

    # 输出目录
    output_dir = "/home/yanai-lab/ma-y/work/assignment/assignment_3/output/5"
    os.makedirs(output_dir, exist_ok=True)

    # 随机噪声与标签
    noise  = torch.randn(batch_size, nz, 1, 1, device=device)
    labels = torch.randint(0, n_classes, (batch_size,), device=device)

    # 生成假图像
    with torch.no_grad():
        fake_images = netG(noise, labels)  # 形状 (B, nc, 64, 64)

    # 将像素范围 [-1,1] 映射到 [0,1]
    fake_images_vis = (fake_images + 1.0) / 2.0

    # 2.1 保存整批拼图
    save_image(
        fake_images_vis,
        os.path.join(output_dir, "fake_images_grid.png"),
        nrow=4, normalize=False
    )

    # 2.2 保存每张单独图像
    for i, img in enumerate(fake_images_vis):
        save_image(
            img,
            os.path.join(output_dir, f"fake_{i:03d}.png")
        )

    # 判别器给出分数并保存
    with torch.no_grad():
        scores = netD(fake_images, labels)  # 形状 (B,)
    # 保存为 PyTorch 张量文件
    torch.save(scores, os.path.join(output_dir, "scores.pt"))

    print(f"已将生成结果和判别分数保存到：{output_dir}")
