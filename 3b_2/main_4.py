# cam_gradcam.py

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# 如果要使用 GPU，把下面两行改成 True 并确保 CUDA 可用
USE_CUDA = False
device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# 1. 加载预训练 ResNet50 模型（自带 GAP + FC 结构） :contentReference[oaicite:0]{index=0}
model = models.resnet50(pretrained=True).to(device)
model.eval()

# -------------------------------------------------------------------
# 2. 图像预处理：缩放到 224×224，标准化到 ImageNet 均值/方差 :contentReference[oaicite:1]{index=1}
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])

def load_image(path):
    """加载并预处理图像，返回 Tensor 和原始 NumPy 图片"""
    img = Image.open(path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # (1,3,224,224)
    orig = np.array(img.resize((224,224)))               # 可视化时用
    return img_tensor, orig

# -------------------------------------------------------------------
# 3. CAM 实现：提取最后一层特征图和 FC 层权重，计算加权和热图 
class CAM:
    def __init__(self, model):
        self.model = model
        self.feature_extractor = model.layer4[-1].conv3
        self.features = None
        self.feature_extractor.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features = output.detach()

    def __call__(self, input_tensor, class_idx=None):
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        params = list(self.model.fc.parameters())[0]
        weights = params[class_idx].unsqueeze(-1).unsqueeze(-1)
        cam_map = F.relu((weights * self.features[0]).sum(dim=0))
        cam_map = cam_map.detach().cpu().numpy()
        # 关键：将小尺寸热图放大到与输入图一致
        cam_map = cv2.resize(cam_map, (224,224))
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        return cam_map, class_idx

# -------------------------------------------------------------------
# 4. Grad-CAM 实现：前向+反向，计算梯度权重并生成热图 :contentReference[oaicite:4]{index=4}
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.layer4[-1].conv3
        self.features = None
        self.gradients = None
        # 注册前向 & 反向钩子
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.features = output.detach()  # (1,2048,H,W)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()  # (1,2048,H,W)

    def __call__(self, input_tensor, class_idx=None):
        logits = self.model(input_tensor)                          # (1,1000)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        # 反向传播梯度到目标类别
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)           # :contentReference[oaicite:5]{index=5}
        # 计算梯度全局平均
        weights = self.gradients.mean(dim=(2,3), keepdim=True)     # (1,2048,1,1) :contentReference[oaicite:6]{index=6}
        # 加权特征并 ReLU
        cam = F.relu((weights * self.features).sum(dim=1))[0]
        cam_map = cam.detach().cpu().numpy()

        cam_map = cv2.resize(cam_map, (224,224))
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        return cam_map, class_idx

# -------------------------------------------------------------------
# 5. 可视化与保存函数
def visualize(orig, cam_map, out_path):
    """将热图叠加到原图并保存"""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + orig * 0.6
    cv2.imwrite(out_path, overlay[..., ::-1])  # BGR->RGB

# -------------------------------------------------------------------
# 6. 主函数示例
if __name__ == "__main__":
    img_path = "/home/yanai-lab/ma-y/work/assignment/assignment_3/image/2/input03.jpg" 
    input_tensor, orig_img = load_image(img_path)

    # CAM 可视化
    cam = CAM(model)
    cam_map, cls = cam(input_tensor)
    visualize(orig_img, cam_map, f"/home/yanai-lab/ma-y/work/assignment/assignment_3/output/cam_{cls}.jpg")

    # Grad-CAM 可视化
    gradcam = GradCAM(model)
    grad_map, cls2 = gradcam(input_tensor)
    visualize(orig_img, grad_map, f"/home/yanai-lab/ma-y/work/assignment/assignment_3/output/gradcam_{cls2}.jpg")

    print(f"Saved CAM for class {cls} -> cam_{cls}.jpg")
    print(f"Saved Grad-CAM for class {cls2} -> gradcam_{cls2}.jpg")
