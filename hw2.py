# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 18:32
# @Author  : AI悦创
# @FileName: hw2.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 加载已经训练好的词向量模型
model_w2v = Word2Vec.load('w2v.model')


# 将文本转化为词向量
def text_to_vec(text_list, model_w2v):
    vec_list = [model_w2v.wv[word] if word in model_w2v.wv.index_to_key else np.zeros(model_w2v.vector_size) for word in
                text_list]
    return np.mean(vec_list, axis=0)


# 加载数据，并将分词结果转为词向量
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data.iloc[:, 1:].apply(lambda x: [i for i in x if isinstance(i, str)], axis=1)
    data['vec'] = data.text.apply(lambda x: text_to_vec(x, model_w2v))
    return data[['label', 'vec']]


train_ns = load_data('data2/train_ns.csv')
val_ns = load_data('data2/val_ns.csv')
test_ns = load_data('data2/test_ns.csv')

# 将标签进行编码
le = LabelEncoder()
train_ns['label_encoded'] = le.fit_transform(train_ns.label)
val_ns['label_encoded'] = le.transform(val_ns.label)
test_ns['label_encoded'] = le.transform(test_ns.label)

# 获取训练，验证，测试数据和标签
X_train = np.stack(train_ns.vec.values)
y_train = to_categorical(train_ns.label_encoded.values)
X_val = np.stack(val_ns.vec.values)
y_val = to_categorical(val_ns.label_encoded.values)
X_test = np.stack(test_ns.vec.values)
y_test = to_categorical(test_ns.label_encoded.values)


# 定义并训练模型
def create_train_model(activation='relu', dropout_rate=0.5, l2_norm=0.01):
    model = Sequential()
    model.add(
        Dense(128, activation=activation, kernel_regularizer=regularizers.l2(l2_norm), input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64)
    return model


# 创建并保存模型
model_relu = create_train_model(activation='relu')
model_relu.save('nn_relu.model')

model_sigmoid = create_train_model(activation='sigmoid')
model_sigmoid.save('nn_sigmoid.model')

model_tanh = create_train_model(activation='tanh')
model_tanh.save('nn_tanh.model')


# 对测试集进行预测并计算准确率
def test_model_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    return accuracy_score(y_true, y_pred)


accuracy_relu = test_model_accuracy(model_relu, X_test, y_test)
accuracy_sigmoid = test_model_accuracy(model_sigmoid, X_test, y_test)
accuracy_tanh = test_model_accuracy(model_tanh, X_test, y_test)

print('ReLU model accuracy:', accuracy_relu)
print('Sigmoid model accuracy:', accuracy_sigmoid)
print('Tanh model accuracy:', accuracy_tanh)
