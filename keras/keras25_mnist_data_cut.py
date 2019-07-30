#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf

# # seed 값 설정
# seed = 0
# numpy.random.seed(seed)
# tf.set_random_seed(seed)

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train[:300]
Y_train = Y_train[:300]
# import matplotlib.pyplot as plt
# digit = X_train[33]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show() # 이미지를 보여주면 코드가 멈춤. 창을 닫아야 다음 코드가 진행됨


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
print(Y_train.shape)    # (60000, )
print(Y_test.shape)
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape)    # (60000, 10) 원-핫 인코딩(One-hot encoding)...
print(Y_test.shape)     # 10은 10개로 데이터가 분류된다는 거

# one-hot(원핫)인코딩이란? 단 하나의 값만 True이고 나머지는 모두 False인 인코딩을 말한다. 
# 즉, 1개만 Hot(True)이고 나머지는 Cold(False)이다. 
# 예를들면 [0, 0, 0, 0, 1]이다. 5번째(Zero-based 인덱스이므로 4)만 1이고 나머지는 0이다. 

# 여러 개로 분류하는 분류 문제에서 
# 타켓 클래스 값을 정수로 사용할 수도 있고 One-Hot 인코딩한 값을 사용할 수도 있다.
# 연산 편의성, 다양한 모델 적용, 정확도 향상 등의 이유로 인코딩/디코딩한다.


# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)


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


# loss='categorical_crossentropy': 분류 모델에서는 이거로 써야 함
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 최적화 설정
# MODEL_DIR = './model/'
# if not os.path.exists(MODEL_DIR):
#     os.mkdir(MODEL_DIR)

# modelpath='./model/{epoch:02d}-{val_loss:.4f}.hdf5'
# checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_b....)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=100, batch_size=10, verbose=1,
                    callbacks=[early_stopping_callback])

# 테스트 정확도 출력
# 분류 모델에서는 Accuracy를 쓰는게 좋음
print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))
