# -*- coding: utf-8 -*-
import numpy as np

'''输入序列的时间步数'''
timesteps = 100
'''输入特征空间维数'''
input_features = 32
'''输出特征空间维数'''
output_features = 64

'''仅作为示例用，随机生成'''
inputs = np.random.random((timesteps, input_features))

'''初始状态：全零向量'''
state_t = np.zeros((output_features, ))

'''创建随机的权重矩阵'''
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features, ))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    '''将输出保存到列表中'''
    successive_outputs.append(output_t)
    '''更新网络状态，用于下一个时间步'''
    state_t = output_t

final_output_sequence = np.stack(successive_outputs, axis=0)
