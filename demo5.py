# -*- coding: utf-8 -*-
# @Time    : 2023/6/30 00:18
# @Author  : AI悦创
# @FileName: demo4.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/

# 加载所需的库
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Activation

# 加载数据
def load_data(file_name):
    df = pd.read_csv(file_name, header=None)
    df.columns = ['label'] + list(range(df.shape[1] - 1))
    df['text'] = df[df.columns[1:]].apply(
        lambda x: ' '.join(x.dropna().astype(str)),
        axis=1
    )
    df = df[['label', 'text']]
    df['text'] = df['text'].apply(lambda x: x.split())
    return df

# 加载word2vec模型并准备嵌入矩阵
def load_word2vec_model(model_name, tokenizer):
    model = Word2Vec.load(model_name)
    word_vectors = model.wv
    vocabulary_size = len(word_vectors.key_to_index) + 1
    embedding_matrix = np.zeros((vocabulary_size, word_vectors.vector_size))
    for word, i in tokenizer.word_index.items():
        if word in word_vectors.key_to_index:
            embedding_matrix[i] = word_vectors.get_vector(word)
    return embedding_matrix, vocabulary_size

# 定义模型
def define_model(hidden_activation, vocabulary_size, embedding_matrix, dropout_rate):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size,
                        output_dim=embedding_matrix.shape[1],
                        input_length=max_length,
                        weights=[embedding_matrix],
                        trainable=False))
    model.add(Dense(64, activation=hidden_activation, kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 加载数据，训练模型并保存
train_data = load_data('data2/train.csv')
val_data = load_data('data2/val.csv')
test_data = load_data('data2/test.csv')

# Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['text'])
max_length = max([len(s) for s in train_data['text']])

embedding_matrix, vocabulary_size = load_word2vec_model('w2v.model', tokenizer)

X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['text']), maxlen=max_length, padding='post')
X_val = pad_sequences(tokenizer.texts_to_sequences(val_data['text']), maxlen=max_length, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['text']), maxlen=max_length, padding='post')

# Convert labels to one-hot encoding
le = LabelEncoder()
y_train = le.fit_transform(train_data['label'])
y_val = le.transform(val_data['label'])
y_test = le.transform(test_data['label'])

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Now create and train your model
model_results = {}

for activation in ['relu', 'sigmoid', 'tanh']:
    for dropout_rate in [0.3, 0.5, 0.7]:
        model = define_model(activation, vocabulary_size, embedding_matrix, dropout_rate)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)
        model.save('nn_' + activation + '.model')
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        model_results[(activation, dropout_rate)] = accuracy

for k, v in model_results.items():
    print(f'Activation: {k[0]}, Dropout rate: {k[1]}, Accuracy: {v}')
