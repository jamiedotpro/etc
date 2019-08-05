# 선형회귀
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 1. 데이터
x_data = np.array([[0,0], [1,0], [0,1], [1,1]])
y_data = np.array([0,1,1,0])  # 위 데이터 and한 결과


# 2. 모델
model = Sequential()
model.add(Dense(8, input_shape=(2, ), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# 3. 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=2000, batch_size=1, verbose=1)


# 4. 평가 예측
x_test = np.array([[0,0], [1,0], [0,1], [1,1]])
y_test = np.array([0,1,1,0])
loss, acc = model.evaluate(x_test, y_test)

# sigmoid 결과를 분류로 출력함
y_predict = model.predict_classes(x_test)
print('loss: ', loss)
print('acc: ', acc)
print(y_predict)
