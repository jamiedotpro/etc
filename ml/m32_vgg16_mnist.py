from keras.applications import VGG16
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0], 28 * 28 * 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28 * 28 * 1).astype('float32') / 255

# 이미지를 3채널로 변환
x_train = np.dstack([x_train] * 3)
x_test = np.dstack([x_test] * 3)

x_train = x_train.reshape(-1, 28, 28, 3)
x_test = x_test.reshape (-1, 28, 28, 3)

# VGG16에서 요구하는대로 이미지 크기를 48 * 48로 조정
from keras.preprocessing.image import img_to_array, array_to_img
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in x_test])

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

conv_base = VGG16(weights='imagenet', include_top=False,
                    input_shape=(48, 48, 3))

from keras.models import Sequential
from keras.layers import Dense, Flatten

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(monitor='loss', patience=10)

# 모델의 실행
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=30, batch_size=200, verbose=1,
                    callbacks=[early_stopping_callback])

# 테스트 정확도 출력
# 분류 모델에서는 Accuracy를 쓰는게 좋음
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\n Test loss: %.4f' % (test_loss))
print('\n Test Accuracy: %.4f' % (test_acc))
