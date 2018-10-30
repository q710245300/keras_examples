# -*- coding: utf-8 -*-
#张量点积
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape[0])

#z = np.dot(x, y)

#向量点积
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

#矩阵-向量点积

def naive_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i], y)
    return z

#矩阵点积
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0], y.shape[1])
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z