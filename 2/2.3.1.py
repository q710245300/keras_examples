# -*- coding: utf-8 -*-
#逐元素运算
#relu
def naive_relu(x):
    assert len(x.shape) == 2

    #避免覆盖输入张量
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x
#加法
def naive_add(x, y):
    '''assert相当于条件判断语句'''
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

import numpy as np

#z = x + y
#z = np.maximum(z, 0.)