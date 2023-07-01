# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 20:20
# @Author  : AI悦创
# @FileName: demo3.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# Load the pre-trained Word2Vec model
w2v_model = Word2Vec.load('w2v.model')


# Function to convert a sentence into a vector by taking the average of the word vectors
def sentence_to_vector(sentence):
    words = sentence.split(',')
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    return np.mean(vectors, axis=0)


# Function to load and prepare the dataset
def load_dataset(filename):
    # Load the data
    df = pd.read_csv(filename, header=None, names=['label', 'text'])

    # Convert the text and labels to the format we need
    df['text'] = df['text'].apply(sentence_to_vector)
    df['label'] = df['label'].map({'positive': 1, 'negative': 0})

    # Remove any rows with missing data
    df = df.dropna()

    # Separate the features and the labels
    X = np.stack(df['text'].values)
    y = df['label'].values

    return X, y


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2


def create_model(activation='relu', dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(128, activation=activation, input_shape=(w2v_model.vector_size,), kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
