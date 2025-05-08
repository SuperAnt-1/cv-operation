import os
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


# --- 1. 加载预训练 VGG16 并切换到评估模式 ---
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.eval()  # 关闭 Dropout，BatchNorm 使用训练时的统计量 :contentReference[oaicite:0]{index=0}

# --- 2. 读取并预处理图像 ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
])
img = Image.open("/home/yanai-lab/ma-y/work/assignment/assignment_3/image/2/input03.jpg").convert("RGB")
input_tensor = preprocess(img).unsqueeze(0).requires_grad_(True)

# --- 3. 前向计算并反向传播到输入 ---
output = model(input_tensor)
pred_class = output.argmax(dim=1)
score = output[0, pred_class]
score.backward()

# --- 4. 提取显著性图并归一化 ---
saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
saliency = saliency.squeeze().cpu().numpy()

# --- 5. 绘制并保存到文件 ---
plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(saliency, cmap='hot')

# 保存路径自定为 output/saliency_map.png
plt.savefig("/home/yanai-lab/ma-y/work/assignment/assignment_3/output/bp.png",
            bbox_inches='tight',    # 去除多余白边
            pad_inches=0.1)         # 缩放边距 :contentReference[oaicite:1]{index=1}
plt.close()  # 关闭当前图，释放内存
