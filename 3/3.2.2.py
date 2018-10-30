# -*- coding: utf-8 -*-
#models.Sequential类，层的线性堆叠
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(784, )))
model.add(layers.Dense(10, activation='softmax'))

#函数式API定义，可进行有向无环图网络的堆叠
input_tensor = layers.Input(shape=(784, ))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)

#接下来的步骤相同
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics=['accuracy'])
models.fit(input_tensor, target_tensor, batch_size=128, epochs=10)
