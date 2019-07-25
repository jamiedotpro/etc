from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# minmax scaler: 0이상 1이하의 값으로 데이터를 확대/축소한다.
# standard scaler: 평균을 0으로 구해서 분산인 -값과 +값을 구할 수 있다.
# 모델의 데이터를 모아주는 역할을 함
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

from sklearn.preprocessing import StandardScaler, MinMaxScaler
#scaler = StandardScaler()
scaler = MinMaxScaler()

# fit. 처음 한번만 실행하여 min/max 기준값을 정함. scaler.fit(x_train) 한번만
# x_train, x_test, x_val은 transform 하면됨.
# y는 할 필요 없음
# y는 트레이닝하는 데이터가 아니라 예측할 값이기 때문
scaler.fit(x)           # x에 있는 값을 기준으로 0~1 MinMaxScaler 진행
x = scaler.transform(x) # 실제 0~1로 변경
print(x)

'''
print('x.shape: ', x.shape)
print('y.shape: ', y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
print('x.reshape: ', x.shape)


# 2. 모델 구성
model = Sequential()
# LSTM(10, ...) 에서 10은 출력으로 Dense임
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

model.summary()


model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=500, verbose=0)

#x_input = array([11,12,13])
x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1))  # 1행 3열에 하나씩 잘라서

yhat = model.predict(x_input)
print(yhat)
'''