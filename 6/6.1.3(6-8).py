# -*- coding: utf-8 -*-
'''由于没有足够的数据进行Embedding层的训练，所以用预先训练好的词嵌入数据库'''
import os

'''对标签进行预处理'''
imdb_dir = './data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        '''后四个字符'''
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

'''对数据进行分词'''
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

'''评论最长限制到100个字，剩下的不要了'''
maxlen = 100
'''在200个样本上训练'''
training_samples = 200
'''在10000个样本上验证'''
validation_samples = 10000
'''只考虑数据集中最常见的前10000个单词'''
max_words = 10000

'''只考虑数据集中最常见的前10000个单词'''
tokenizer = Tokenizer(num_words=max_words)
'''构建单词索引'''
tokenizer.fit_on_texts(texts)
'''将字符串转化为整数索引组成的列表'''
sequences = tokenizer.texts_to_sequences(texts)

'''保存单词索引'''
word_index = tokenizer.word_index
print(word_index)
print('Found %s unique tokens.' %len(word_index))

'''将整数列表转化为二维整数张量(sequences, maxlen)'''
data = pad_sequences(sequences, maxlen=maxlen)
'''将list转化为数组'''
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

'''将数据划分为训练集和验证集，注意打乱顺序'''
'''用于创建等差数列'''
indices = np.arange(data.shape[0])
'''打乱顺序'''
np.random.shuffle(indices)
'''用打乱的索引打乱训练数据'''
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]



'''下载GloVe词嵌入'''
'''预处理词嵌入'''
glove_dir = './data/glove.6B'

'''字典'''
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
'''每一行是一个参数'''
for line in f:
    values = line.split()
    '''第一维是单词'''
    word = values[0]
    '''其余99维是参数'''
    cofes = np.asarray(values[1:], dtype='float32')
    '''构建词向量索引字典'''
    embeddings_index[word] = cofes
f.close()

print('Found %s word vectors.' % len(embeddings_index))

'''对于imdb的10000个最常见单词准备GloVe词嵌入矩阵'''
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
'''{word, i}'''
for word, i in word_index.items():
    '''只保留前10000个常见单词'''
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        '''同时这个单词在GloVe词嵌入文件中存在'''
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

'''模型定义'''
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

'''将预训练的词嵌入加载到Embedding层中'''
model.layers[0].set_weights([embedding_matrix])
'''GloVe层相当于一个训练好的Embedding层，所以加载后不用训练Embedding层'''
model.layers[0].trainable = False

'''训练与评估模型'''
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

'''绘制结果'''
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
'''plot(横坐标， 纵坐标，线类型， 线名字)'''
plt.plot(epochs, acc, 'bo', label='Trainning acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Trainning and Validation accuracy')
plt.legend()

'''新建窗口'''
plt.figure()

plt.plot(epochs, loss, 'bo', label='Trainning loss')
plt.plot(epochs, val_acc, 'b', label='Validation loss')
plt.title('Trainning and Validation loss')
plt.legend()

plt.show()

'''对比测试，当同样数据应用在不使用预训练词嵌入的情况下的准确度和损失函数'''
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
               loss='binary_crossentropy',
               metrics=['acc'])

history = model.fit(x_train, y_train,
                     epochs=10,
                     batch_size=32,
                     validation_data=(x_val, y_val))

#plt.clf()

plt.plot(epochs, history.history['acc'], 'bo', label='Training acc without GloVe')
plt.plot(epochs, history.history['val_acc'], 'b', label='Validation acc without GloVe')
plt.title('Training and Validation acc without GloVe')
plt.legend()

plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss without GloVe')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss without GloVe')
plt.title('Training and Validation loss without GloVe')
plt.legend()
plt.show()

'''对测试集进行分词'''
test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen)
y_test = np.asarray(labels)

model.load_weights('pre_trained_glove_model.h5')
results = model.evaluate(x_test, y_test)
print(results)
