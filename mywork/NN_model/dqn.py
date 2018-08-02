from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def NN_build(available_actions_count):
    model = Sequential()
    #two conv layer
    model.add(Conv2D(32, 
        kernel_size=(8,8),
        activation='relu',
        input_shape=(60,108,3)
    ))
    model.add(Conv2D(64,
        kernel_size=(4,4),
        activation='relu'

    ))
    model.add(Flatten())
    #full connect
    model.add(Dense(available_actions_count))

    return model
