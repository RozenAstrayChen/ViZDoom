from settings import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import random

Collect_Mode = False
TestData_Mode = False
Train_Mode = True





if Collect_Mode:

    x_table,tableName = findAllFile()

    SaveXModule(x_table,tableName)
    y_current = Load_Yvalue('./y_value.txt')
    y_train = Reshape_Y(y_current)
    SaveYModule(y_train)

if TestData_Mode:
    x = np.load('./Xepsoide1.npy')
    y = np.load('./Yepsoide1.npy')


if Train_Mode:

    x = np.load('./Xepsoide1.npy')
    y = np.load('./Yepsoide1.npy')
    # 84*84 = 7056
    #x_resize = x.reshape(100, 7056)

    x_train , y_train = x[:90],y[:90]
    x_test , y_test = x[90:],y[90:]

    # got randomInteger
    randomInteger = random.randint(0, len(x_test))
    x_test_model = x_test[randomInteger]

    x_train = x_train.reshape(90,7056)
    x_test = x_test.reshape(10,7056)


    #unknow
    #y_train = to_categorical(y_train, num_classes=3)
    #y_test = to_categorical(y_test, num_classes=3)

    model = Sequential()
    model.add(Dense(32, input_dim=7056))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    #optimzers
    rms = optimizers.RMSprop(lr=0.01, epsilon=1e-8, rho=0.9)

    model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=10, epochs=10, shuffle=True)

    score = model.evaluate(x_test, y_test, batch_size=10)
    h = model.predict_classes(x_test_model.reshape(1, -1),batch_size=1)

    #y_pred = model.predict(x_test)
    print ('loss:\t', score[0], '\naccuracy:\t', score[1])
    print ('\nclass:\t', h)
    plt.imshow(x_test_model,cmap='gray')
    plt.show()
