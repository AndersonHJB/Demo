# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 20:09
# @Author  : AI悦创
# @FileName: demo2.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
import torch
from torchtext.legacy import data
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from torch.nn import functional as F

# Step 1: Load and process the data
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype=torch.float)

fields = [('label', LABEL), ('text', TEXT)]

# You will need to replace the file paths with the paths of your own files
train_data, valid_data, test_data = data.TabularDataset.splits(
    path='your_path_here',
    train='train_ns.csv',
    validation='val_ns.csv',
    test='test_ns.csv',
    format='csv',
    fields=fields,
    skip_header=False
)

# Load Word2Vec model
w2v_model = Word2Vec.load("w2v.model")

# Build vocab and load word2vec vectors
TEXT.build_vocab(train_data, max_size=25000, vectors=w2v_model)
LABEL.build_vocab(train_data)

# Prepare the data iterators
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort_within_batch=True
)


# Step 2: Create and train the models
class FFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        hidden = self.dropout(self.fc1(embedded))
        return self.fc2(hidden)


EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
DROPOUT = 0.5

vocab_size = len(TEXT.vocab)

# create models
model_relu = FFNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
model_sigmoid = FFNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
model_tanh = FFNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)

# choose the optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

models = [(model_relu, F.relu), (model_sigmoid, torch.sigmoid), (model_tanh, torch.tanh)]

# place the models and loss function to device
for model, _ in models:
    model = model.to(device)
criterion = criterion.to(device)


# function to calculate the accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# function for training
def train(model, iterator, optimizer, criterion, activation):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(activation(predictions), batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# function for evaluation
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 5

best_valid_loss = float('inf')

for model, activation in models:
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, activation)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

# Step 3: Test the models
for model, _ in models:
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
