import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# 붓꽃 데이터 읽어 들이기
dir_path = os.getcwd()
iris_data_file = os.path.join(dir_path, 'etc/data/iris.csv')
iris_data = pd.read_csv(iris_data_file, encoding='utf-8', header=None,
                        names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name'])

# print(iris_data)
# print(iris_data.shape)
# print(type(iris_data))

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
# loc는 레이블로 분리
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = iris_data.loc[:, 'Name']

# iloc는 컬럼의 순서로 분리
# x2 = iris_data.iloc[:, 0:4]
# y2 = iris_data.iloc[:, 4]

# print(x.shape)
# print(y.shape)

SepalLength = np.array(iris_data['SepalLength'])
SepalWidth = np.array(iris_data['SepalWidth'])
PetalLength = np.array(iris_data['PetalLength'])
PetalWidth = np.array(iris_data['PetalWidth'])

SepalLength = np.reshape(SepalLength, (-1, 1))
SepalWidth = np.reshape(SepalWidth, (-1, 1))
PetalLength = np.reshape(PetalLength, (-1, 1))
PetalWidth = np.reshape(PetalWidth, (-1, 1))

x = np.concatenate([SepalLength, SepalWidth, PetalLength, PetalWidth], axis=1)
y = np.array(iris_data['Name'])

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

print(x_train.shape)
print(x_test.shape)


# 문자열 데이터여서 One-hot encoding 할 수 있게 숫자로 바꿔줌
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)
# print(encoder.classes_)
# print(encoder.inverse_transform([0, 1, 2]))
# print(y_train)

# One-hot encoding...
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)


# 학습하기
model = Sequential()
model.add(Dense(8, input_shape=(4, ), activation='relu'))
model.add(Dense(3, activation='softmax'))

# loss='categorical_crossentropy': 분류 모델에서는 이거로 써야 함
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='loss', patience=10)

# 모델의 실행
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=500, batch_size=10, verbose=1,
                    callbacks=[early_stopping_callback])

# 테스트 정확도 출력
# 분류 모델에서는 Accuracy를 쓰는게 좋음
print('\n Test Accuracy: %.4f' % (model.evaluate(x_test, y_test)[1]))
y_predict = model.predict_classes(x_test)

print(y_predict)
print(encoder.inverse_transform(y_predict))
