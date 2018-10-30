# -*- coding: utf-8 -*-
'''单词级的one-hot编码'''
import numpy as np

'''初始数据：每个样本是列表的一个元素(本例中的样本是一个句子，但也可以是一整篇文档)'''
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

'''构建数据中所有标记的索引'''
token_index = {}
for sample in samples:
    '''使用split对句子进行分词，具体应用时还要从样本中去掉标点'''
    for word in sample.split():
        if word not in token_index:
            '''为这个单词指定唯一索引，从1开始'''
            token_index[word] = len(token_index) + 1

'''对样本进行分词。只考虑每个样本前max_length个单词'''
max_length = 10

results = np.zeros(shape=(len(samples),
                   max_length,
                   max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.

print(token_index.get('The'))
print(results)