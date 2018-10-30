# -*- coding: utf-8 -*-
#张量变形
#reshape()
import numpy as np

x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.shape)
x = x.reshape(6, 1)
print(x)
#矩阵转置
x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape)