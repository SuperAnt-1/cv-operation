import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# ---------------------
# Configuration
# ---------------------
OUTPUT_DIR = '/home/yanai-lab/ma-y/work/assignment/assignment_4/output/1'
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------
# Model Definition
# ---------------------
class Encoder(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 32->16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16->8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8->4
            nn.ReLU(True)
        )
        self.fc = nn.Linear(256*4*4, bottleneck_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super().__init__()
        self.fc = nn.Linear(bottleneck_dim, 256*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4->8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8->16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 16->32
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        recon = self.deconv(x)
        return recon

class Autoencoder(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super().__init__()
        self.encoder = Encoder(bottleneck_dim)
        self.decoder = Decoder(bottleneck_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

# ---------------------
# Training Loop
# ---------------------

def train_autoencoder():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(root='/home/yanai-lab/ma-y/work/assignment/CIFAR',train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = Autoencoder(bottleneck_dim=256).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS+1):
        epoch_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {epoch_loss:.6f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'autoencoder.pth'))
    return model

# ---------------------
# Feature Extraction & Clustering
# ---------------------

def cluster_bottleneck(model):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Randomly select 100 samples
    indices = random.sample(range(len(test_dataset)), 100)
    subset = Subset(test_dataset, indices)
    loader = DataLoader(subset, batch_size=100, shuffle=False)

    imgs, _ = next(iter(loader))
    imgs = imgs.to(DEVICE)

    # Extract bottleneck features
    with torch.no_grad():
        features = model.encoder(imgs).cpu().numpy()  # shape (100, 256)

    # Clustering
    for k in [5, 10]:
        kmeans = KMeans(n_clusters=k, random_state=0)
        clusters = kmeans.fit_predict(features)
        # Save assignments
        df = pd.DataFrame({
            'index': indices,
            'cluster': clusters
        })
        df.to_csv(os.path.join(OUTPUT_DIR, f'clusters_k{k}.csv'), index=False)

        # Make directories for cluster visuals
        cluster_dir = os.path.join(OUTPUT_DIR, f'cluster_images_k{k}')
        os.makedirs(cluster_dir, exist_ok=True)

        # Save a grid of images per cluster
        for c in range(k):
            imgs_c = imgs[clusters == c]
            if len(imgs_c) == 0:
                continue
            grid = vutils.make_grid(imgs_c[:10], nrow=5, normalize=True, scale_each=True)
            vutils.save_image(grid, os.path.join(cluster_dir, f'cluster_{c}.png'))

        print(f"Clustering done for k={k}. Results saved to {cluster_dir} and CSV.")

# ---------------------
# Main
# ---------------------

def main():
    model = train_autoencoder()
    cluster_bottleneck(model)

if __name__ == '__main__':
    main()
