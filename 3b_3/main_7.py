import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# -------------------------
# 工具函数：加载图像并预处理
# -------------------------
def load_image(path, max_size=256):
    img = Image.open(path).convert('RGB')
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return transforms.ToTensor()(img).unsqueeze(0)

# -------------------------
# Gram 矩阵计算
# -------------------------
def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(c, h * w)
    return torch.mm(features, features.t()).div(c * h * w)

# -------------------------
# Total Variation Loss
# -------------------------
def tv_loss(x):
    return torch.mean(torch.abs(x[..., :-1, :] - x[..., 1:, :])) + \
           torch.mean(torch.abs(x[..., :, :-1] - x[..., :, 1:]))

# -------------------------
# 模型封装：提取内容/风格特征
# -------------------------
class VGGFeatures(nn.Module):
    def __init__(self, content_layers, style_layers):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        # 将所有 ReLU 改为非 in-place，避免就地修改计算图
        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.ReLU):
                vgg[i] = nn.ReLU(inplace=False)
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg.eval()
        self.content_layers = content_layers
        self.style_layers = style_layers

    def forward(self, x):
        cf, sf = {}, {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.content_layers:
                cf[name] = x
            if name in self.style_layers:
                sf[name] = gram_matrix(x)
        return cf, sf

# -------------------------
# 风格迁移主流程
# -------------------------
def run_style_transfer(content_path, style_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载图像
    content   = load_image(content_path).to(device)
    style     = load_image(style_path).to(device)
    generated = content.clone().requires_grad_(True).to(device)

    # 提取哪些层的内容和风格特征
    content_layers = ['21']                   # conv4_2
    style_layers   = ['0','5','10','19','28'] # conv1_1 ~ conv5_1

    model = VGGFeatures(content_layers, style_layers).to(device)
    tgt_c, _ = model(content)
    _,    tgt_s = model(style)

    # 使用 Adam 优化器，更稳定
    optimizer = optim.Adam([generated], lr=1e-2)

    # 各损失权重
    content_weight = 1e0
    style_weight   = 1e4
    tv_weight      = 1e-6

    for step in range(1, 301):
        optimizer.zero_grad()
        cf, sf = model(generated)
        # 内容损失
        loss_c = content_weight * torch.mean((cf['21'] - tgt_c['21'])**2)
        # 风格损失
        loss_s = 0
        for l in style_layers:
            loss_s += torch.mean((sf[l] - tgt_s[l])**2)
        loss_s *= style_weight
        # 总变差损失
        loss_tv = tv_weight * tv_loss(generated)
        loss = loss_c + loss_s + loss_tv

        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_([generated], max_norm=1.0)
        optimizer.step()
        # 每次迭代后 clamp 像素
        with torch.no_grad():
            generated.clamp_(0, 1)

        if step % 50 == 0:
            print(f"Step {step}/300 — content: {loss_c.item():.2f}, style: {loss_s.item():.2f}, tv: {loss_tv.item():.2e}")

    # 最终 clamp
    with torch.no_grad():
        generated.clamp_(0, 1)

    # 转为图像张量并放大 2 倍
    out_tensor = generated.squeeze(0).cpu()  # [C,H,W]
    # 插值放大
    out_up = F.interpolate(out_tensor.unsqueeze(0), scale_factor=2, mode='bicubic', align_corners=False)
    out_up = out_up.squeeze(0).permute(1, 2, 0)  # [H*2, W*2, C]
    # 转 0-255 并保存
    out_img = (out_up * 255).clamp(0, 255).byte().numpy()
    save_path = '/home/yanai-lab/ma-y/work/assignment/assignment_3/output/7/output_up2x.png'
    Image.fromarray(out_img).save(save_path)
    print(f"放大×2 后的结果已保存为 {save_path}")

if __name__ == '__main__':
    run_style_transfer(
        '/home/yanai-lab/ma-y/work/assignment/assignment_3/image/7/content.jpg',
        '/home/yanai-lab/ma-y/work/assignment/assignment_3/image/7/style.jpg'
    )
