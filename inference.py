# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 22:30
# @Author  : AI悦创
# @FileName: inference.py.py
# @Software: PyCharm
# @Blog    ：https://bornforthis.cn/
# import sys
# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# from gensim.models import Word2Vec
#
#
# def inference(sentence, model_name):
#     # 加载模型
#     model = load_model(model_name)
#     w2v_model = Word2Vec.load('w2v.model')
#
#     # 转换句子为词向量
#     sentence_vec = [w2v_model.wv[word] for word in sentence.split() if word in w2v_model.wv]
#     sentence_vec = pad_sequences([sentence_vec], padding='post')
#
#     # 预测
#     prediction = model.predict(sentence_vec)
#
#     # 返回预测结果
#     return 'positive' if prediction[0] > 0.5 else 'negative'
#
#
# if __name__ == "__main__":
#     sentence = sys.argv[1]
#     model_name = sys.argv[2]
#     print(inference(sentence, model_name))
# import numpy as np
# from keras.models import load_model
# from gensim.models import Word2Vec
# import jieba
#
#
# def load_w2v_model(model_path):
#     return Word2Vec.load(model_path)
#
#
# def sentence_to_vec(sentence, w2v_model):
#     words = jieba.lcut(sentence)
#     vec = np.zeros((25, 100))
#     for i, word in enumerate(words[:25]):
#         if word in w2v_model.wv:  # add this line to handle words not in the vocabulary
#             vec[i, :] = w2v_model.wv[word]
#     return vec
#
#
# def inference(sentence, model_name):
#     # 加载模型和词向量模型
#     model = load_model(model_name)
#     w2v_model = load_w2v_model('w2v.model')
#
#     sentence_vec = sentence_to_vec(sentence, w2v_model)
#     sentence_vec = np.expand_dims(sentence_vec, axis=0)
#
#     prediction = model.predict(sentence_vec)
#     if prediction[0][0] > 0.5:
#         return 'positive'
#     else:
#         return 'negative'
#
#
# if __name__ == '__main__':
#     import sys
#
#     with open(sys.argv[1], 'r') as f:
#         sentences = f.readlines()
#     for sentence in sentences:
#         print(inference(sentence, sys.argv[2]))
import numpy as np
import sys
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec


def inference(sentence, model_name):
    # 加载模型
    model = load_model(model_name)
    w2v_model = Word2Vec.load('w2v.model')

    # 转换句子为词向量
    sentence_vec = [w2v_model.wv[word] for word in sentence.split() if word in w2v_model.wv]
    if len(sentence_vec) == 0:
        sentence_vec = [np.zeros(w2v_model.vector_size)]
    sentence_vec = pad_sequences([sentence_vec], padding='post', dtype='float32', maxlen=25)

    # 预测
    prediction = model.predict(sentence_vec)

    # 返回预测结果
    return 'positive' if prediction[0] > 0.5 else 'negative'


if __name__ == "__main__":
    sentence = sys.argv[1]
    model_name = sys.argv[2]
    print(inference(sentence, model_name))
