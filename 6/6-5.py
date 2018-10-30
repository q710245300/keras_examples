# -*- coding: utf-8 -*-
'''将一个Embedding层实例化'''
from keras.datasets import imdb
from keras import preprocessing
'''作为特征的单词个数'''
max_features = 10000
'''在这么多单词后截断文本(这些单词都属于前max_features个最常见的单词)'''
maxlen = 20

'''将数据加载为整数列表'''
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

'''将整数列表转换成形状为(samples, maxlen)的二维整数张量'''
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

'''在IMDB数据上使用Embedding层和分类器'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
'''对于这10000个单词，网络将对每个词都学习一个8维嵌入'''
'''指定Embedding层的最大输入长度，以便于后面将输出展平'''
model.add(Embedding(10000, 8, input_length=maxlen))

'''Flatten层展开张量'''
model.add(Flatten())

'''添加分类器'''
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)