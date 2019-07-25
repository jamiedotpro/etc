from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print('x.shape: ', x.shape)
print('y.shape: ', y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
print('x.reshape: ', x.shape)


# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

model.summary()


model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000, verbose=0)
# verbose=0으로 하면 계산과정이 출력안함. 1은 자세히 표시되고 기본값. 2는 간략히 표시됨. 다른 값은 epochs 진행만 보임

#x_input = array([11,12,13])
x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1))  # 1행 3열에 하나씩 잘라서

yhat = model.predict(x_input, verbose=1)
# verbose=0으로 하면 계산과정이 출력안함. 1은 계산 시간 출력됨
print(yhat)
