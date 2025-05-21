import json
import pickle
import os
import re
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ----------------------------------
# 词汇表构建
# ----------------------------------
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def tokenizer(self, text):
        # 使用正则表达式进行简单分词
        return re.findall(r"\w+", text.lower())

    def build_vocabulary(self, annotations_file):
        with open(annotations_file, 'r') as f:
            captions = json.load(f)['annotations']
        freq = Counter()
        for ann in captions:
            tokens = self.tokenizer(ann['caption'])
            freq.update(tokens)
        idx = len(self.itos)
        for word, count in freq.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.stoi.get(tok, self.stoi['<UNK>']) for tok in tokens]

# ----------------------------------
# COCO 数据集定义
# ----------------------------------
class CocoCaptionDataset(Dataset):
    def __init__(self, image_dir, annotations_file, vocab, transform=None, max_len=50):
        self.image_dir = image_dir
        self.transform = transform
        self.max_len = max_len

        with open(annotations_file, 'r') as f:
            data = json.load(f)
        img_map = {img['id']: img['file_name'] for img in data['images']}
        self.samples = [(os.path.join(image_dir, img_map[ann['image_id']]), ann['caption'])
                        for ann in data['annotations']]

        if isinstance(vocab, str):
            with open(vocab, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tokens = [self.vocab.stoi['<SOS>']] + self.vocab.numericalize(caption) + [self.vocab.stoi['<EOS>']]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            tokens[-1] = self.vocab.stoi['<EOS>']
        length = len(tokens)
        tokens += [self.vocab.stoi['<PAD>']] * (self.max_len - length)
        return image, torch.tensor(tokens), length

# ----------------------------------
# 模型定义
# ----------------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, images):
        features = self.resnet(images).view(images.size(0), -1)
        features = self.norm(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths+1, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return self.linear(outputs)

# ----------------------------------
# 主函数：词表、数据加载、模型、训练流程
# ----------------------------------
def main():
    # 超参
    freq_threshold = 5
    max_len = 50
    batch_size =16
    embed_size = 256
    hidden_size = 512
    num_epochs = 5
    learning_rate = 1e-3

    # 1. 构建词表
    vocab = Vocabulary(freq_threshold)
    ann_file = '/home/yanai-lab/ma-y/work/assignment/COCO/annotations/captions_train2014.json'  # 用户指定的路径
    vocab.build_vocabulary(ann_file)
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary size: {len(vocab.stoi)}")

    # 2. 数据加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CocoCaptionDataset(
        image_dir='/home/yanai-lab/ma-y/work/assignment/COCO/train2014',
        annotations_file=ann_file,
        vocab=vocab,
        transform=transform,
        max_len=max_len
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Total samples: {len(dataset)}")

    # 3. 模型初始化
    vocab_size = len(vocab.stoi)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = EncoderCNN(embed_size).to(device)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

    # 4. 定义损失与优化器
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<PAD>'])
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.norm.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # 5. 训练循环
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0

        # 用 tqdm 包裹 loader，desc 显示当前 epoch 信息
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")
        for images, captions, lengths in loop:
            images = images.to(device)
            captions = captions.to(device)
            targets = nn.utils.rnn.pack_padded_sequence(
                captions, lengths, batch_first=True, enforce_sorted=False
            )[0]

            # 前向
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            outputs = nn.utils.rnn.pack_padded_sequence(
                outputs[:, 1:, :], lengths, batch_first=True, enforce_sorted=False
            )[0]

            # 计算与优化
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 每个 batch 后更新进度条后缀，显示实时 loss
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {avg_loss:.4f}")



    # 6. 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, 'models/cnn_lstm_caption.pth')
    print("Training complete. Model saved to models/cnn_lstm_caption.pth")

if __name__ == '__main__':
    main()
