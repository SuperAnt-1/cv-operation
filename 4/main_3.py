#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet101

# 确保项目根目录在搜索路径里
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    # 1. 构建带 auxiliary 分类器的 FCN-ResNet101
    model = fcn_resnet101(
        pretrained=False,
        num_classes=21,
        aux_loss=True
    )
    # 2. 加载权重
    snapshot = os.path.join("snapshots", "fcn_resnet101_coco.pth")
    if not os.path.isfile(snapshot):
        raise FileNotFoundError(f"找不到权重文件：{snapshot}")
    state = torch.load(snapshot, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # 3. 准备输入图像
    img_path = "/home/yanai-lab/ma-y/work/assignment/assignment_4/input/3/2.jpg"
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"找不到输入图像：{img_path}")
    origin = Image.open(img_path).convert("RGB")

    # 4. 预处理
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    input_tensor = preprocess(origin).unsqueeze(0)

    # 5. 推理
    with torch.no_grad():
        output = model(input_tensor)["out"]
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 6. 保存灰度掩码
    gray_path = "example_mask.png"
    Image.fromarray(pred.astype("uint8")).save(gray_path)
    print(f"灰度掩码已保存：{gray_path}")

    # 7. 彩色调色板（Pascal VOC 21 类）
    palette = np.array([
        [  0,   0,   0],  # 0=background
        [128,   0,   0],  # 1=aeroplane
        [  0, 128,   0],  # 2=bicycle
        [128, 128,   0],  # 3=bird
        [  0,   0, 128],  # 4=boat
        [128,   0, 128],  # 5=bottle
        [  0, 128, 128],  # 6=bus
        [128, 128, 128],  # 7=car
        [ 64,   0,   0],  # 8=cat
        [192,   0,   0],  # 9=chair
        [ 64, 128,   0],  # 10=cow
        [192, 128,   0],  # 11=diningtable
        [ 64,   0, 128],  # 12=dog
        [192,   0, 128],  # 13=horse
        [ 64, 128, 128],  # 14=motorbike
        [192, 128, 128],  # 15=person
        [  0,  64,   0],  # 16=pottedplant
        [128,  64,   0],  # 17=sheep
        [  0, 192,   0],  # 18=sofa
        [128, 192,   0],  # 19=train
        [  0,  64, 128],  # 20=tvmonitor
    ], dtype=np.uint8)

    # 8. 生成并保存彩色掩码
    color_mask = palette[pred]  # H×W×3
    color_path = "mask_color.png"
    Image.fromarray(color_mask).save(color_path)
    print(f"彩色掩码已保存：{color_path}")

    # 9. 叠加展示并保存
    mask_img = Image.fromarray(color_mask).resize(origin.size)
    overlay = Image.blend(origin, mask_img, alpha=0.6)
    overlay_path = "overlay.png"
    overlay.save(overlay_path)
    print(f"叠加图已保存：{overlay_path}")

    # 10. 打印像素分布
    classes, counts = np.unique(pred, return_counts=True)
    print("预测类别索引：", classes)
    print("对应像素数：", counts)

if __name__ == "__main__":
    main()
