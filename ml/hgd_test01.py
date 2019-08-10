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

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# 예측은 원데이터로 함
x_train_original = x_train
y_train_original = y_train
x_test_original = x_test
y_test_original = y_test

# 분석은 300개를 가지고 아래서 증폭시켜서 함
x_train = x_train[:300]
y_train = y_train[:300]

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

scaler_original = StandardScaler()
x_train_original = x_train_original.reshape(len(x_train_original) * 32 * 32, 3)
x_test_original = x_test_original.reshape(len(x_test_original) * 32 * 32, 3)
scaler_original.fit(x_train_original)
x_train_original = scaler_original.transform(x_train_original)
x_test_original = scaler_original.transform(x_test_original)

# dnn
# x_train = x_train.reshape(int(x_train.shape[0] / 32 / 32), 32 * 32 * 3)
# x_test = x_test.reshape(int(x_test.shape[0] / 32 / 32), 32 * 32 * 3)
# x_train_original = x_train_original.reshape(int(x_train_original.shape[0] / 32 / 32), 32 * 32 * 3)
# x_test_original = x_test_original.reshape(int(x_test_original.shape[0] / 32 / 32), 32 * 32 * 3)

# cnn
x_train = x_train.reshape(int(x_train.shape[0] / 32 / 32), 32, 32, 3)
x_test = x_test.reshape(int(x_test.shape[0] / 32 / 32), 32, 32, 3)
x_train_original = x_train_original.reshape(int(x_train_original.shape[0] / 32 / 32), 32, 32, 3)
x_test_original = x_test_original.reshape(int(x_test_original.shape[0] / 32 / 32), 32, 32, 3)

# 범주형으로 변환
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
y_train_original = np_utils.to_categorical(y_train_original, NB_CLASSES)
y_test_original = np_utils.to_categorical(y_test_original, NB_CLASSES)


# 하이퍼 파라미터 최적화
def build_network_dnn(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(32 * 32 * 3,), name='input')
    x = Dense(128, activation='relu', name='hidden1')(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    prediction = Dense(NB_CLASSES, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def build_network_cnn_063(keep_prob=0.0, optimizer='adam'):
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), name='input')
    x = Conv2D(32, (3, 3), padding='same', name='hidden1')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(60, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(keep_prob)(x)

    prediction = Dense(NB_CLASSES, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model
    # {'optimizer': 'adadelta', 'keep_prob': 0.2, 'batch_size': 10}
    # Test Accuracy: 0.6345

def build_network_cnn_065(keep_prob=0.0, optimizer='adam'):
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), name='input')
    x = Conv2D(32, (3, 3), padding='same', name='hidden1')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(60, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(keep_prob)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(keep_prob)(x)

    prediction = Dense(NB_CLASSES, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model
    # {'optimizer': 'adam', 'keep_prob': 0.0, 'batch_size': 10}
    # Test Accuracy: 0.6509

def build_network_cnn(keep_prob=0.0, optimizer='adam'):
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), name='input')
    x = Conv2D(32, (3, 3), padding='same', name='hidden1')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(keep_prob)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(keep_prob)(x)

    prediction = Dense(NB_CLASSES, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [1,5,10,20]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.0, 0.4, 5)
    return{'batch_size':batches, 'optimizer':optimizers, 'keep_prob':dropout}

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함
# model = KerasClassifier(build_fn=build_network_dnn, verbose=1)
model = KerasClassifier(build_fn=build_network_cnn, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=-1, cv=3, verbose=1)
                            # 작업이 10회 수행, 3겹 교차검증 사용

# 이미지 개수를 늘린 후에 fit
# 이미지 생성기
from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(
    # 지정값 범위 내에서 회전, 가로, 세로, 변경..
    rotation_range=20,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True
)

# 이미지 개수 늘리기
add_img_cnt = 200
x_train_cnt = len(x_train)
for j in range(add_img_cnt):
    for i in range(x_train_cnt):    
        img = x_train[i, :, :, :]
        img = img.reshape((1,) + img.shape)

        check = 0
        for batch in data_generator.flow(img, batch_size=1):
            x_train = np.vstack((x_train, batch))
            y_train = np.vstack((y_train, y_train[i]))
            k = batch
            check += 1
            if check == 1:
                break
        
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)

search.fit(x_train, y_train)

print(search.best_params_)

print('score: ', search.score(x_test, y_test))
print('predict: \n', search.predict(x_test))


#------------------------------
# 위에서 hyperparameter 튜닝한 값으로 원데이터로 결과를 예측

model_dic = search.best_params_

# model = build_network_dnn(model_dic['keep_prob'], model_dic['optimizer'])
model = build_network_cnn(model_dic['keep_prob'], model_dic['optimizer'])
model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

# 모델의 실행
history = model.fit(x_train_original, y_train_original,
                    validation_data=(x_test_original, y_test_original),
                    epochs=500, batch_size=model_dic['batch_size'], verbose=1,
                    callbacks=[early_stopping])

# 테스트 정확도 출력
# 분류 모델에서는 Accuracy를 쓰는게 좋음
print('\n Test Accuracy: %.4f' % (model.evaluate(x_test_original, y_test_original)[1]))


# 이미지 개수 *2 일 때 결과
# {'optimizer': 'adadelta', 'keep_prob': 0.0, 'batch_size': 1}
# score:  0.2181
# Test Accuracy: 0.1062