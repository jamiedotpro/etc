from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization


from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import BatchNormalization

from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import os


# CIFAR_10은 3채널로 구성된 32*32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# 상수 정의
BATCH_SIZE = 128
NB_EPOCH = 40
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()


# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# ...Scaler 사용해서 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()

# print(x_train)
# scaler 사용을 위해 reshape
# 50000, 32, 32, 3
x_train = x_train.reshape(len(x_train) * 32 * 32, 3)
x_test = x_test.reshape(len(x_test) * 32 * 32, 3)



scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# dnn
# x_train = x_train.reshape(int(x_train.shape[0] / 32 / 32), 32 * 32 * 3)
# x_test = x_test.reshape(int(x_test.shape[0] / 32 / 32), 32 * 32 * 3)

# cnn
x_train = x_train.reshape(int(x_train.shape[0] / 32 / 32), 32, 32, 3)
x_test = x_test.reshape(int(x_test.shape[0] / 32 / 32), 32, 32, 3)

# 범주형으로 변환
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

def model_dnn():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(32 * 32 * 3,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    return model


def model_cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(Conv2D(60, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    return model
    # Test Accuracy: 0.6722

def model_cnn2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3,3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    return model

model = model_cnn2()
model.summary()


# loss='categorical_crossentropy': 분류 모델에서는 이거로 써야 함
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping_callback = EarlyStopping(monitor='loss', patience=10, mode='auto')

# 모델의 실행
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=500, batch_size=10, verbose=1,
                    callbacks=[early_stopping_callback])


# 테스트 정확도 출력
# 분류 모델에서는 Accuracy를 쓰는게 좋음
print('\n Test Accuracy: %.4f' % (model.evaluate(x_test, y_test)[1]))
