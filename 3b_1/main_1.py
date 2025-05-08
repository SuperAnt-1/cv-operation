import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 根据要用的 GPU 数量修改


import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



# 选择数据集 10 类或 100 类
num_classes = 10  # 若跑 CIFAR-100 则改为 100

transform = transforms.Compose([
    transforms.Resize(224),               # 兼容预训练网络
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
) if num_classes == 10 else torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform
)

# 根据 GPU 数量调整总 batch size，例：单 GPU 256，2 GPU 512，4 GPU 1024
total_batch_size = 256 * torch.cuda.device_count()

trainloader = DataLoader(
    trainset, batch_size=total_batch_size,
    shuffle=True, num_workers=min(2*torch.cuda.device_count(), 16)
)


# 以 ResNet18 为例
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=None, num_classes=num_classes)


# DataParallel 自动在所有可见 GPU 上复制模型并同步梯度
model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = nn.CrossEntropyLoss()

base_lr        = 0.1
world_size     = torch.cuda.device_count()
scaled_lr      = base_lr * world_size


optimizer = optim.SGD(model.parameters(), scaled_lr, momentum=0.9, weight_decay=5e-4)


num_epochs= 5


# 线性 warmup
warmup_steps = 500
def lr_lambda(step):
    return min((step + 1) / warmup_steps, 1.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)






def train_one_epoch(epoch):
    model.train()                                 # 切换模型到训练模式，启用 Dropout 和 BatchNorm 的训练行为
    start = time.time()                           # 记录当前时间，用于计算本轮训练耗时
    running_loss = 0.0                            # 初始化累积损失值，用于后面求平均损失
    for inputs, labels in trainloader:            # 遍历 DataLoader 中的每个批次（inputs：输入张量，labels：对应标签）
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据搬到指定设备（GPU 或 CPU）
        optimizer.zero_grad()                     # 清空上一步计算残留的梯度，防止梯度累加
        outputs = model(inputs)                   # 前向计算，得到模型在当前批次上的预测结果
        loss = criterion(outputs, labels)         # 计算损失函数值（交叉熵等）
        loss.backward()                           # 反向传播，自动计算每个参数的梯度并累加到 .grad
        optimizer.step()                          # 梯度下降一步，更新模型参数
        scheduler.step()                          # 更新学习率调度器（如 warmup 或 decay）
        running_loss += loss.item() * inputs.size(0)  # 将本批次的总损失累加（loss.item() 是平均损失，乘以 batch 大小得到总损失）

    elapsed = time.time() - start                 # 计算本轮训练耗时（秒）
    avg_loss = running_loss / len(trainloader.dataset)  # 计算整个训练集上的平均损失
    print(f"[GPUの数={torch.cuda.device_count()}] Epoch {epoch}: "
          f"時間 {elapsed:.2f}s, 平均損失 {avg_loss:.4f}")  # 打印当前用 GPU 数、轮次、耗时与平均损失
    return elapsed                                # 返回本轮耗时，便于外部记录与对比


# 示例：只跑一个 epoch 来比较时间
if __name__ == "__main__":
   for epoch in range(1, num_epochs+1):
        train_one_epoch(epoch)