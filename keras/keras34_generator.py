#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf

# 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train[:300]
Y_train = Y_train[:300]


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
print(Y_train.shape)    # (60000, )
print(Y_test.shape)
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape)    # (60000, 10) 원-핫 인코딩(One-hot encoding)...
print(Y_test.shape)     # 10은 10개로 데이터가 분류된다는 거


# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
# model.add(Conv2D(128, (3,3), activation='relu'))
# model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))  # 분류 모델을 사용하려면 마지막에 무조건 activation='softmax'로 설정해야함
# Test Accuracy: 0.9386

# loss='categorical_crossentropy': 분류 모델에서는 이거로 써야 함
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


# 이미지 생성기
from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(
    # 지정값 범위 내에서 회전, 가로, 세로, 변경..
    rotation_range=20,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True
)
# model.fit_generator 는 model.fit을 대체한다.
# fit보다 fit_generator이 느리다. 이미지 개수를 늘려서 학습하기 때문
# fit_generator 할때만 증폭된 개수로 fit이 되는 것으로, 실제 데이터는 증가되지 않는다.
model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=32),
                    # 몇배로 증폭시킬 것인가
                    # X_train * steps_per_epoch => 실제로 생성(증폭)되는 이미지 개수
                    steps_per_epoch=200, # len(X_train) // 32,
                    epochs=200,
                    validation_data=(X_test, Y_test),
                    verbose=1,
                    callbacks=[early_stopping_callback])

# 모델의 실행
# history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
#                     epochs=100, batch_size=10, verbose=1,
#                     callbacks=[early_stopping_callback])

# 테스트 정확도 출력
# 분류 모델에서는 Accuracy를 쓰는게 좋음
print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))


# 300개를 6만개로 만들기
# 증폭시켜서 CNN 돌리기

