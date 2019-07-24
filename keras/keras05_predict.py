# 행은 무시되고, 열이 우선이다.


# 1. 데이터
import numpy as np
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x3 = np.array([101, 102, 103, 104, 105, 106])
#x4 = np.array([301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315])
x4 = np.array(range(30, 50))

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    # 순차적으로 내려가는 모델이 Sequential

# 깊이, 개수, epochs, batch_size 순으로 조정
#model.add(Dense(5, input_dim=1, activation='relu')) # input_dim. 입력되는 데이터의 열개수
model.add(Dense(5, input_shape=(1, ), activation='relu'))  # input_shape = (1, ) ==> 행은 모르고, 열이 하나
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=200)

model.summary()

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', acc)

y_predict = model.predict(x4)
print(y_predict)
