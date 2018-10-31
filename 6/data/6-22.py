# -*- coding: utf-8 -*-

'''将SimpleRNN模型应用与IMDB电影评论分类问题'''
'''准备IMDB数据'''
from keras.datasets import imdb
from keras.preprocessing import sequence

'''作为特征单词的个数'''
max_features = 10000
'''在这么多个单词后截断文本(这些单词都属于max_features个最常见单词)'''
maxlen = 500
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences(samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

'''用一个Embedding层和SimpleRNN层来训练一个简单的循环网络'''
from keras.layers import Dense
from keras import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
'''Embedding层输入(max_features, 网络学习后每个词向量所占的维度)， 输出(max_features, sequences_length, 词向量所占的维度)'''
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

'''绘制结果'''
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'r', label='Training_acc')
plt.plot(epochs, val_acc, 'b', label='Validation_acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training_loss')
plt.plot(epochs, val_loss, 'b', label='Validation_loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

'''在第 3 章，处理这个数据集的第一个简单方法得到的测试精度是 88%。不幸的是，
与这个基准相比，这个小型循环网络的表现并不好（验证精度只有 85%）。问题的部分原因在于，
输入只考虑了前 500 个单词，而不是整个序列， 因此， RNN 获得的信息比前面的基准模型更少。
另一部分原因在于， SimpleRNN 不擅长处理长序列，比如文本。'''