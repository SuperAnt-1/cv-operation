# main_7_2.py

import json
import pickle
import os
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ----------------------------------
# 词汇表构建（与训练脚本一致）
# ----------------------------------
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def tokenizer(self, text):
        # 简单正则分词
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
# 模型定义（与训练脚本一致）
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
        # 输出 (batch, 2048)，再映射到 embed_size
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
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths+1, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return self.linear(outputs)

# ----------------------------------
# 辅助函数：加载图片 & 采样生成字幕
# ----------------------------------
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # shape: [1,3,224,224]

def sample_caption(encoder, decoder, image_tensor, vocab, max_len=20, device='cpu'):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        feature = encoder(image_tensor)               # [1, embed]
        inputs = feature.unsqueeze(1)                 # [1,1,embed]
        states = None
        sampled_ids = []
        for _ in range(max_len):
            hiddens, states = decoder.lstm(inputs, states)        # [1,1,hidden]
            outputs = decoder.linear(hiddens.squeeze(1))          # [1, vocab_size]
            _, predicted = outputs.max(1)                         # [1]
            sampled_ids.append(predicted.item())
            if vocab.itos[predicted.item()] == '<EOS>':
                break
            inputs = decoder.embed(predicted).unsqueeze(1)        # [1,1,embed]
        # 转换 id → 单词
        words = []
        for word_id in sampled_ids:
            word = vocab.itos[word_id]
            if word == '<EOS>':
                break
            words.append(word)
    return ' '.join(words)

# ----------------------------------
# 主函数：加载 vocab & 模型，测试单张图像
# ----------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载词表
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)    # 需先定义 Vocabulary 类，才能成功反序列化 :contentReference[oaicite:0]{index=0}

    # 2. 初始化模型并加载权重
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab.stoi)
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
    checkpoint = torch.load('/home/yanai-lab/ma-y/work/assignment/assignment_4/models/cnn_lstm_caption.pth', map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    # 3. 测试示例
    test_image = '/home/yanai-lab/ma-y/work/assignment/assignment_4/input/7/1.jpg'   # 替换为实际路径
    img_tensor = load_image(test_image)
    caption = sample_caption(encoder, decoder, img_tensor, vocab, device=device)
    print(f'Image: {test_image}\nGenerated Caption: {caption}')

if __name__ == '__main__':
    main()
