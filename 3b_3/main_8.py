import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# ---------------------------
# 自定义 Dataset：FlatFolderDataset
# ---------------------------
class FlatFolderDataset(Dataset):
    """
    遍历单层目录下所有指定后缀的图像文件，返回图像和占位标签 0
    """
    def __init__(self, root, transform=None, extensions=('jpg','jpeg','png')):
        self.paths = []
        for ext in extensions:
            self.paths.extend(glob.glob(os.path.join(root, f'*.{ext}')))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0

# ---------------------------
# 模块定义：网络组件
# ---------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True)
    def forward(self, x): return self.bn(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, 3, 1),
            nn.ReLU(),
            ConvLayer(channels, channels, 3, 1)
        )
    def forward(self, x): return x + self.block(x)

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True)
    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode='nearest')
        return self.bn(self.conv(x))

# ---------------------------
# Transformer 网络主体
# ---------------------------
class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.conv3 = ConvLayer(64,128, 3, 2)
        self.res1  = ResidualBlock(128)
        self.res2  = ResidualBlock(128)
        self.res3  = ResidualBlock(128)
        self.res4  = ResidualBlock(128)
        self.res5  = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, 3, 1, upsample=2)
        self.deconv2 = UpsampleConvLayer(64, 32, 3, 1, upsample=2)
        self.deconv3 = ConvLayer(32, 3, 9, 1)
    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.deconv1(y))
        y = self.relu(self.deconv2(y))
        return self.deconv3(y)

# ---------------------------
# 辅助函数：Gram 矩阵 & 特征提取
# ---------------------------
def gram_matrix(y):
    b, ch, h, w = y.size()
    features = y.view(b, ch, h*w)
    G = torch.bmm(features, features.transpose(1,2))
    return G / (ch * h * w)

def extract_features(x, vgg):
    feats = {}
    out = x
    for name, layer in vgg._modules.items():
        out = layer(out)
        if name=='3':  feats['relu1_2'] = out
        if name=='8':  feats['relu2_2'] = out
        if name=='15': feats['relu3_3'] = out
        if name=='22': feats['relu4_3'] = out
    return feats

# ---------------------------
# 主函数：训练流程
# ---------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # 图像预处理 & 数据加载
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    dataset = FlatFolderDataset(
        root='/home/yanai-lab/ma-y/work/assignment/train2014',
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # 网络与 VGG16 特征网络
    transformer = TransformerNet().to(device)
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
    for p in vgg.parameters(): p.requires_grad=False

    # 加载并预计算风格图像 Gram
    style_img = Image.open('/home/yanai-lab/ma-y/work/assignment/assignment_3/image/7/style.jpg').convert('RGB')
    style_tensor = transform(style_img).unsqueeze(0).to(device)
    style_feats = extract_features(style_tensor, vgg)
    style_targets = {l:gram_matrix(style_feats[l]) for l in style_feats}

    # 优化器与超参
    optimizer = optim.Adam(transformer.parameters(), lr=1e-3)
    style_weight, content_weight = 1e5, 1e0
    num_epochs=2; log_interval=500; save_interval=2000
    os.makedirs('/home/yanai-lab/ma-y/work/assignment/assignment_3/output/8', exist_ok=True)

    it=0
    for epoch in range(num_epochs):
        transformer.train()
        for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            it+=1
            x = x.to(device)
            optimizer.zero_grad()
            y = transformer(x)
            # 内容损失
            fy = extract_features(y, vgg)
            fx = extract_features(x, vgg)
            c_loss = content_weight * torch.mean((fy['relu2_2']-fx['relu2_2'])**2)
            # 风格损失
            s_loss = sum(torch.mean((gram_matrix(fy[l])-style_targets[l])**2)
                         for l in style_targets) * style_weight
            loss = c_loss + s_loss
            loss.backward(); optimizer.step()
            if it%log_interval==0:
                print(f"Iter {it}: content={c_loss.item():.2f}, style={s_loss.item():.2f}")
            if it%save_interval==0:
                transformer.eval()
                with torch.no_grad():
                    for i, img in enumerate(y[:4]):
                        transforms.ToPILImage()(img.cpu().clamp(0,1))\
                            .save(f"/home/yanai-lab/ma-y/work/assignment/assignment_3/output/8/iter_{it}_{i}.jpg")
                transformer.train()

    torch.save(transformer.state_dict(), '/home/yanai-lab/ma-y/work/assignment/assignment_3/output/8/transformer.pth')

if __name__=='__main__':
    main()