# -*- coding: utf-8 -*-
'''字符级的one-hot编码'''
import string
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
'''所有可以打印的ASCII字符'''
characters = string.printable
'''使用zip函数快速创建字典'''
token_index = dict(zip(range(1, len(characters) + 1), characters))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.

print(results)
