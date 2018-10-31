# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

'''只返回最后一个时间步的输出'''
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

'''返回完整的状态序列'''
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()

'''为了提高网络的能力，将多个循环层叠加，只是所有的中间层都要返回完整的输出序列'''
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()
