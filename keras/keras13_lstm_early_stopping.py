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

from keras.callbacks import EarlyStopping
# loss: 오차 손실, patience: 지정값 이상 loss값이 변화가 없으면 멈춘다.
# loss 값이 줄어들수록(0에 가까울수록) 최적의 weight 값을 찾게된다.
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x, y, epochs=10000, verbose=1, callbacks=[early_stopping])

#x_input = array([11,12,13])
x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1))  # 1행 3열에 하나씩 잘라서

yhat = model.predict(x_input, verbose=1)
print(yhat)
