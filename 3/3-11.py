# -*- coding: utf-8 -*-
from keras.datasets import imdb

#仅保留训练数据中前10000个最长出现的单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#print(train_data[0])
#print(len(train_data))
#print(test_labels[0])
print(max([max(sequence) for sequence in train_data]))

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    '''创建一个形状为(len(sequences), dimension)的零矩阵'''
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results
'''数据向量化'''
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
'''标签向量化'''
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

'''构建网络'''
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

'''从新训练一个模型'''
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)
print(model.predict(x_test))