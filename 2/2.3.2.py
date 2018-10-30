# -*- coding: utf-8 -*-
'''如果没有歧义的话,较小的张量会被广播(broadcast),以匹配较大张量的形状。广播包含
以下两步。
(1) 向较小的张量添加轴(叫作广播轴),使其 ndim 与较大的张量相同。
(2) 将较小的张量沿着新轴重复,使其形状与较大的张量相同。'''

#广播
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x

import numpy as np

x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))

z = np.maximum(x, y)