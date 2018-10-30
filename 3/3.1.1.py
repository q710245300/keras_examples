# -*- coding: utf-8 -*-
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784, )))
#可以自动连接上一层输出的32维向量作为输入，所以不用设置input_shape
model.add(layers.Dense(32))