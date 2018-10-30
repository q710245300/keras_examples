# -*- coding: utf-8 -*-
import numpy as np
'''长短一致的list组成的ndarry是张量，不一致的不是，'''
a = [1, 2]
b = [4, 5, 6]
print(type(a))
x = np.array([a, b])
print(x.ndim)
print(x.shape)
print(x[0])
print(x)


from keras.datasets import imdb

#仅保留训练数据中前10000个最长出现的单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data)
print(len(train_data))
print(train_data.shape)
print(train_data.ndim)
print(test_data[0])