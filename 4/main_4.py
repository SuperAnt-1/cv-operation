#!/usr/bin/env python3
# fasterrcnn_save.py

import os
import torch
from PIL import Image, ImageDraw
from torchvision import transforms, models

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    return img, to_tensor(img).unsqueeze(0)  # 返回 PIL.Image 和 [1,3,H,W] Tensor

def draw_and_save(pil_img, boxes, scores, labels, out_path, score_thresh=0.5):
    draw = ImageDraw.Draw(pil_img)
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1-10), f"{label}:{score:.2f}", fill="red")
    pil_img.save(out_path)
    print(f"Saved annotated image to {out_path}")

def main():
    # 输入输出路径
    img_path = "/home/yanai-lab/ma-y/work/assignment/assignment_4/input/4/1.jpg"
    out_dir  = "/home/yanai-lab/ma-y/work/assignment/assignment_4/output/4"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Faster R-CNN.jpg")

    # 设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device).eval()

    # 读取图像并推理
    pil_img, tensor_img = load_image(img_path)
    tensor_img = tensor_img.to(device)
    with torch.no_grad():
        outputs = model(tensor_img)[0]  # 取第一个结果

    # 提取检测结果
    boxes  = outputs["boxes"].cpu()
    scores = outputs["scores"].cpu()
    labels = outputs["labels"].cpu()

    # 绘制 & 保存
    draw_and_save(pil_img, boxes, scores, labels, out_path)

if __name__ == "__main__":
    main()
