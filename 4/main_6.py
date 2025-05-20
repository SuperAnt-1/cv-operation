import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os, pickle, numpy as np
from PIL import Image

# 1. 自定义 CIFAR-10 数据集（同你之前的实现）

class CIFAR10Custom(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        batch_files = [f'data_batch_{i}' for i in range(1,6)] if train else ['test_batch']
        for fname in batch_files:
            path = os.path.join(data_dir, fname)
            with open(path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.labels += entry['labels']
        self.data = np.concatenate(self.data, axis=0)
        # CIFAR 存储格式为 (N, 3072)，reshape 成 (N,3,32,32)
        self.data = self.data.reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].transpose(1,2,0)   # (H,W,C)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# 2. 数据变换与 DataLoader
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
data_dir = '/home/yanai-lab/ma-y/work/assignment/CIFAR/cifar-10-batches-py'
train_ds = CIFAR10Custom(data_dir, train=True,  transform=transform)
test_ds  = CIFAR10Custom(data_dir, train=False, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=4)

# 3. Vision Transformer 模型（同你之前的 ViT 定义）
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768,
                 depth=12, heads=12, mlp_dim=3072, num_classes=10):
        super().__init__()
        num_patches = (img_size//patch_size)**2
        self.patch_embed = nn.Conv2d(in_ch, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1,embed_dim))
        encoder = nn.TransformerEncoderLayer(d_model=embed_dim,
                                             nhead=heads,
                                             dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder,
                                                num_layers=depth)
        self.to_cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                   # (B,embed,H',W')
        x = x.flatten(2).transpose(1,2)           # (B,patches,embed)
        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_tokens, x), dim=1)     # (B,patches+1,embed)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.to_cls(x[:,0])

# 4. 训练与验证循环，并打印结果
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = ViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
epochs = 5

for epoch in range(1, epochs+1):
    # --- 训练 ---
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    avg_loss = running_loss / len(train_loader.dataset)

    # --- 验证 ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
    val_acc = 100 * correct / len(test_loader.dataset)

    # 打印每个 epoch 的结果
    print(f"Epoch {epoch:>2}/{epochs} → Loss: {avg_loss:.4f}, Val Acc: {val_acc:6.2f}%")
