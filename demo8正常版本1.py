# -*- coding: utf-8 -*-
# @Time    : 2023/7/1 11:32
# @Author  : AI悦创
# @FileName: demo8正常版本1.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder


def load_data(file_name, w2v_model):
    data = pd.read_csv(file_name, header=None)
    data.columns = ["label"] + ["word" + str(i) for i in range(1, data.shape[1])]
    data = data.dropna(axis=1, how="all")
    data["text"] = data.iloc[:, 1:].apply(lambda row: ' '.join(row.dropna().values.astype(str)), axis=1)
    data = data[["label", "text"]]

    # tokenize and pad sequences
    max_length = np.max(data["text"].apply(lambda x: len(x.split())))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data["text"])
    sequences = tokenizer.texts_to_sequences(data["text"])
    word_index = tokenizer.word_index
    data_features = pad_sequences(sequences, maxlen=max_length)

    # prepare embeddings
    embedding_dim = w2v_model.vector_size
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    # convert labels to integer values
    le = LabelEncoder()
    labels = le.fit_transform(data["label"])

    return data_features, labels, max_length, embedding_matrix


w2v_model = Word2Vec.load("w2v.model")

train_data, train_labels, max_length, embedding_matrix = load_data('data2/train.csv', w2v_model)
val_data, val_labels, _, _ = load_data('data2/val.csv', w2v_model)
test_data, test_labels, _, _ = load_data('data2/test.csv', w2v_model)


def create_and_train_model(activation_function, train_data, train_labels, val_data, val_labels, max_length,
                           embedding_matrix):
    embedding_dim = embedding_matrix.shape[1]
    num_words = embedding_matrix.shape[0]

    model = Sequential()
    model.add(Dense(256, activation=activation_function, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10)

    return model


activation_functions = ["relu", "sigmoid", "tanh"]

for activation_function in activation_functions:
    model = create_and_train_model(activation_function, train_data, train_labels, val_data, val_labels, max_length,
                                   embedding_matrix)
    model.save(f"nn_{activation_function}.model")
