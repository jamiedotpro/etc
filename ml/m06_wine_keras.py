import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

# 데이터 읽어 들이기
dir_path = os.getcwd()
winequality_white_file = os.path.join(dir_path, 'etc/data/winequality-white.csv')
wine = pd.read_csv(winequality_white_file, sep=';', encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop('quality', axis=1)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

# 숫자가 일치하지 않아서 한번 거치고 원핫인코딩함
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, test_size=0.2)


# 원활한 훈련을 위해 훈련 데이터 스케일 조정
scaler = StandardScaler()
# scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 학습하기
model = Sequential()
# model.add(Dense(64, input_shape=(11, ), activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# acc:  0.9132653066090175

# model.add(Dense(32, input_shape=(11, ), activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# acc:  0.9204081627787376

# model.add(Dense(50, input_shape=(11, ), activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# acc:  0.9234693875118177

# model.add(Dense(50, input_shape=(11, ), activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# acc:  0.9255102035950641


# model.add(Dense(80, input_shape=(11, ), activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# acc:  0.9357142852277172

model.add(Dense(100, input_shape=(11, ), activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()



# 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=10, verbose=1,
            validation_split=0.2, callbacks=[early_stopping])


# 4. 평가
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss: ', loss)
print('acc: ', acc)

