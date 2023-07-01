# -*- coding: utf-8 -*-
# @Time    : 2023/6/30 00:48
# @Author  : AI悦创
# @FileName: demo6.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
# -*- coding: utf-8 -*-
# @Time    : 2023/6/30 00:18
# @Author  : AI悦创
# @FileName: demo4.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
# 第一步：加载所需的库
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Activation
from keras.utils import to_categorical


# 第二步：加载数据
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


# 第三步：加载word2vec模型并准备嵌入矩阵
def load_word2vec_model(model_name):
    model = Word2Vec.load(model_name)
    word_vectors = model.wv
    vocabulary_size = len(word_vectors.key_to_index) + 1
    embedding_matrix = np.zeros((vocabulary_size, word_vectors.vector_size))
    for word, i in tokenizer.word_index.items():
        if word in word_vectors.key_to_index:
            embedding_matrix[i] = word_vectors.get_vector(word)
    return embedding_matrix, vocabulary_size


# 第四步：定义模型
def define_model(hidden_activation, dropout_rate, vocabulary_size, embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size,
                        output_dim=embedding_matrix.shape[1],
                        input_length=max_length,
                        weights=[embedding_matrix],
                        trainable=False))
    model.add(Dense(64, activation=hidden_activation, kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='softmax')) # 两个节点的Dense层，使用softmax
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy']) # 使用categorical_crossentropy作为损失函数
    return model


# 第五步：加载数据，训练模型并保存
train_data = load_data('data2/train.csv')
val_data = load_data('data2/val.csv')

# Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['text'])
max_length = max([len(s) for s in train_data['text']])

embedding_matrix, vocabulary_size = load_word2vec_model('w2v.model')

X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['text']), maxlen=max_length, padding='post')
X_val = pad_sequences(tokenizer.texts_to_sequences(val_data['text']), maxlen=max_length, padding='post')

# Convert labels to one-hot encoding
le = LabelEncoder()
y_train = le.fit_transform(train_data['label'])
y_val = le.transform(val_data['label'])

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# Now create and train your model

for activation in ['relu', 'sigmoid', 'tanh']:
    for dropout_rate in [0.2, 0.5, 0.8]:
        model = define_model(activation, dropout_rate, vocabulary_size, embedding_matrix)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)
        model.save('nn_' + activation + str(dropout_rate) + '.model')
