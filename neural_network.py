# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 18:45
# @Author  : AI悦创
# @FileName: neural_network.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
import os

# 加载 Word2Vec 模型
w2v_model = Word2Vec.load("w2v.model")

# 创建数据集类
# 创建数据集类
class AmazonDataset(Dataset):
    def __init__(self, csv_file, w2v_model):
        self.dataframe = pd.read_csv(csv_file, header=None, names=["label", "text"])
        self.w2v_model = w2v_model
        self.labelencoder = LabelEncoder().fit(['negative', 'positive'])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx, 0]
        text = self.dataframe.iloc[idx, 1:].dropna().tolist()
        vector = np.mean([self.w2v_model.wv[word] for word in text if word in self.w2v_model.wv], axis=0)
        return self.labelencoder.transform([label])[0], vector


# 创建模型类
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_function):
        super(FeedForwardNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation_function = activation_function
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation_function(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据加载器
train_dataset = AmazonDataset("data2/train_ns.csv", w2v_model)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = AmazonDataset("data2/val_ns.csv", w2v_model)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# 创建模型、优化器和损失函数
activation_functions = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}
hidden_size = 64
output_size = 2

for name, activation_function in activation_functions.items():
    model = FeedForwardNN(w2v_model.vector_size, hidden_size, output_size, activation_function).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)  # 使用 L2 正则化
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    for epoch in range(10):
        total_loss = 0
        for labels, vectors in train_dataloader:
            labels = labels.to(device)
            vectors = vectors.float().to(device)
            optimizer.zero_grad()
            outputs = model(vectors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

    # 保存模型
    torch.save(model.state_dict(), f"nn_{name}.model")
