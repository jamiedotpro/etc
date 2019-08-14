from keras.applications import VGG16
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np

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

# 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ...Scaler 사용해서 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()

x_train = x_train.reshape(x_train.shape[0], IMG_ROWS * IMG_COLS * IMG_CHANNELS)
x_test = x_test.reshape(x_test.shape[0], IMG_ROWS * IMG_COLS * IMG_CHANNELS)

scaler.fit(x_train)
X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, IMG_ROWS, IMG_COLS, IMG_CHANNELS)
x_test = x_test.reshape (-1, IMG_ROWS, IMG_COLS, IMG_CHANNELS)

# VGG16에서 요구하는대로 이미지 크기를 48 * 48로 조정
from keras.preprocessing.image import img_to_array, array_to_img
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in x_test])

# 범주형으로 변환
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

conv_base = VGG16(weights='imagenet', include_top=False,
                    input_shape=(48, 48, 3))

from keras.models import Sequential
from keras.layers import Dense, Flatten

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(NB_CLASSES, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(monitor='loss', patience=10)

# 모델의 실행
history = model.fit(x_train, y_train, validation_split=VALIDATION_SPLIT,# validation_data=(x_test, y_test),
                    epochs=30, batch_size=200, verbose=1,
                    callbacks=[early_stopping_callback])

# 테스트 정확도 출력
# 분류 모델에서는 Accuracy를 쓰는게 좋음
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\n Test loss: %.4f' % (test_loss))
print('\n Test Accuracy: %.4f' % (test_acc))
