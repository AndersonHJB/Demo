# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 19:12
# @Author  : AI悦创
# @FileName: demo1.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.vocab import Vectors
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score

# 加载Word2Vec模型
w2v_model = Word2Vec.load('w2v.model')

# 定义Field
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype=torch.float)

# 加载数据
train_data = data.TabularDataset(
    path='train.csv',
    format='csv',
    fields=[('label', LABEL), ('text', TEXT)],
    skip_header=True
)
valid_data = data.TabularDataset(
    path='val.csv',
    format='csv',
    fields=[('label', LABEL), ('text', TEXT)],
    skip_header=True
)
test_data = data.TabularDataset(
    path='test.csv',
    format='csv',
    fields=[('label', LABEL), ('text', TEXT)],
    skip_header=True
)

# 使用训练集构建词汇表
TEXT.build_vocab(train_data, vectors=Vectors('w2v.model', cache='./'))
LABEL.build_vocab(train_data)


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, activation_function):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_function = activation_function

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        if self.activation_function == 'relu':
            act_func = nn.ReLU()
        elif self.activation_function == 'sigmoid':
            act_func = nn.Sigmoid()
        elif self.activation_function == 'tanh':
            act_func = nn.Tanh()
        else:
            raise ValueError('Invalid activation function')
        hidden = act_func(self.fc1(embedded))
        output = nn.functional.log_softmax(self.fc2(hidden), dim=1)
        return output


# 定义训练函数和测试函数
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = accuracy_score(batch.label, predictions)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = accuracy_score(batch.label, predictions)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def run_model(activation_function, model_path):
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    N_EPOCHS = 5
    best_valid_loss = float('inf')

    model = Net(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT, activation_function)

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)  # L2 regularization
    criterion = nn.CrossEntropyLoss()

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)


run_model('relu', 'nn_relu.model')
run_model('sigmoid', 'nn_sigmoid.model')
run_model('tanh', 'nn_tanh.model')
