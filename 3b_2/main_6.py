#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unconditional DCGAN 训练脚本：基于 PyTorch 官方 DCGAN 教程
应用于 UECFOOD100 数据集，无条件图像生成
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

# ---------------------------
# 1. 超参数与路径
# ---------------------------
z_dim      = 100             # 潜在向量维度
ngf        = 64              # 生成器通道基数
df         = 64              # 判别器通道基数
nc         = 3               # 图像通道数 (RGB)
batch_size = 128             # 批量大小
epochs     = 50              # 训练轮次
lr         = 0.0002          # 学习率
beta1      = 0.5             # Adam beta1

# 数据集路径
data_root  = "/home/yanai-lab/ma-y/work/assignment/UECFOOD/UECFOOD100"
output_dir = "/home/yanai-lab/ma-y/work/assignment/assignment_3/output/6"
os.makedirs(output_dir, exist_ok=True)

# 启用 cuDNN benchmark
cudnn.benchmark = True
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2. 权重初始化函数
# ---------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ---------------------------
# 3. 模型定义 (无条件 DCGAN)
# ---------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, ngf, nc):
        super().__init__()
        self.main = nn.Sequential(
            # 输入：Z，输出大小 ngf*8 x 4 x 4
            nn.ConvTranspose2d(z_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # 状态：(ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # -> (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc, df):
        super().__init__()
        self.main = nn.Sequential(
            # 输入：nc x 64 x 64 -> df x 32 x 32
            nn.Conv2d(nc, df, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (df*2) x 16 x 16
            nn.Conv2d(df, df*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df*2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (df*4) x 8 x 8
            nn.Conv2d(df*2, df*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df*4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (df*8) x 4 x 4
            nn.Conv2d(df*4, df*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df*8),
            nn.LeakyReLU(0.2, inplace=True),
            # -> 1 x 1 x 1
            nn.Conv2d(df*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1)

# 实例化网络并初始化权重
netG = Generator(z_dim=z_dim, ngf=ngf, nc=nc).to(device)
netD = Discriminator(nc=nc, df=df).to(device)
netG.apply(weights_init)
netD.apply(weights_init)

# 多卡并行
if torch.cuda.device_count() > 1:
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

# ---------------------------
# 4. 数据加载
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
dataset = datasets.ImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

# ---------------------------
# 5. 损失与优化器
# ---------------------------
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 固定噪声用于可视化
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

# ---------------------------
# 6. 训练循环
# ---------------------------
for epoch in range(1, epochs+1):
    for i, (real_images, _) in enumerate(dataloader, 1):
        real_images = real_images.to(device)
        bsz = real_images.size(0)

        # 真实标签为1，假标签为0
        real_labels = torch.full((bsz,), 1.0, device=device)
        fake_labels = torch.full((bsz,), 0.0, device=device)

        # ------- 更新判别器 -------
        netD.zero_grad()
        output_real = netD(real_images)
        lossD_real = criterion(output_real, real_labels)

        noise = torch.randn(bsz, z_dim, 1, 1, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, fake_labels)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # ------- 更新生成器 -------
        netG.zero_grad()
        output = netD(fake_images)
        lossG = criterion(output, real_labels)
        lossG.backward()
        optimizerG.step()

        
    # 每轮结束保存一次
    with torch.no_grad():
        fake = netG(fixed_noise).cpu()
        grid = make_grid((fake+1)/2, nrow=8)
        save_image(grid, os.path.join(output_dir, f"epoch{epoch:03d}.png"))

print("训练完成，结果已保存至", output_dir)
