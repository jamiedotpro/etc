from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt

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
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 범주형으로 변환
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)


# 실수형으로 지정하고 정규화
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# ...Scaler 사용해서 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()   # /255 = 0.78 -> StandardScaler 동일
# scaler = MinMaxScaler()    # /255 = 0.78 -> MinMaxScaler 0.76

# print(X_train)
# scaler 사용을 위해 reshape
# 50000, 32, 32, 3
X_train = X_train.reshape(len(X_train) * 32 * 32, 3)
X_test = X_test.reshape(len(X_test) * 32 * 32, 3)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(50000, 32, 32, 3)
X_test = X_test.reshape(10000, 32, 32, 3)


# 신경망 정의
model = Sequential()
model.add(Conv2D(128, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
# 0.7884

# conv2d -> dense 층이 많이 필요 없음
# conv2d 6개 32, 32, 60, 128 노드를 늘리는 식으로
# 레이어 가상 모델링에서 점점 커지는 모습이 있어서 노드를 늘린 편


model.summary()

# 학습
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE,
                    callbacks=[early_stopping_callback, tb_hist])

print('testing...')
score = model.evaluate(X_test, Y_test,)
print('\ntest score:', score[0])
print('test accuracy:', score[1])

# # 모델 저장
# model_json = model.to_json()
# open('cifar10_architecture.json', 'w').write(model_json)
# model.save_weights('cifar10_weights.h5', overwrite=True)


# 사진 한장을 출력(시각화) 확인 후 주석 처리
digit = X_train[33]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show() 


# 히스토리에 있는 모든 데이터 나열
print(history.history.keys())
# 단순 정확도에 대한 히스토리 요약
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
