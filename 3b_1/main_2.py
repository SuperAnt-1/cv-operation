#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def occlusion_sensitivity(img_orig, target_class, preprocess, model, 
                          occ_size=64, stride=8, start=-56, end=247):
    """
    对一张 PIL 图像做遮挡敏感度计算，返回 heatmap 数组 (H, W)。
    """
    ys = list(range(start, end+1, stride))
    xs = list(range(start, end+1, stride))
    heatmap = np.zeros((len(ys), len(xs)), dtype=np.float32)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            img_occ = img_orig.copy()
            x1 = max(x, 0)
            y1 = max(y, 0)
            x2 = min(x + occ_size, img_occ.width)
            y2 = min(y + occ_size, img_occ.height)
            if x1 < x2 and y1 < y2:
                patch = Image.new("RGB", (x2 - x1, y2 - y1), (127, 127, 127))
                img_occ.paste(patch, (x1, y1))
            input_occ = preprocess(img_occ).unsqueeze(0).to(next(model.parameters()).device)
            with torch.no_grad():
                out = model(input_occ)
                prob = F.softmax(out, dim=1)[0, target_class].item()
            heatmap[i, j] = prob

    return heatmap

def save_heatmap(heatmap, save_path):
    """
    把 heatmap 用 jet 颜色映射保存到文件。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='jet', origin='upper')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # 设备与模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # 图片列表
    base_in  = "assignment_3/image/2/input{:02d}.jpg"
    base_out = "assignment_3/output/result_{:02d}.png"

    for idx in range(1, 6):
        img_path  = base_in.format(idx)
        out_path  = base_out.format(idx)

        # 加载并预处理原图
        img_orig = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img_orig).unsqueeze(0).to(device)
        with torch.no_grad():
            out_orig = model(input_tensor)
            probs_orig = F.softmax(out_orig, dim=1)
            target_class = torch.argmax(probs_orig, dim=1).item()

        # 计算 heatmap
        heatmap = occlusion_sensitivity(
            img_orig, target_class, preprocess, model
        )

        # 保存结果
        save_heatmap(heatmap, out_path)
        print(f"[{idx:02d}] 保存热图到 {out_path}")

if __name__ == "__main__":
    main()
