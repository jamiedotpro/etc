import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM


# 1. 데이터
# LSTM은 예측할 값의 범위가 데이터의 최소/최대값 안에 있어야 잘 예측된다.
a = np.array(range(1,11))

size = 5
# LSTM에 넣기 좋은 연속 데이터 만들기
def split_5(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
print('===================================')
print(dataset)

x_train = dataset[:, 0:4]
y_train = dataset[:, 4:]

print(x_train.shape)    # (6, 4)
print(y_train.shape)    # (6, 1)

# LSTM은 몇 개씩 잘라서 작업할 건가를 지정해야하기에 reshape
# x_train = np.reshape(x_train, (6, 4, 1))
x_train = np.reshape(x_train, (len(a)-size+1, 4, 1))

print(x_train.shape)

x_test = np.array([ [[11], [12], [13], [14]], [[12], [13], [14], [15]],
                    [[13], [14], [15], [16]], [[14], [15], [16], [17]] ])
y_test = np.array([15, 16, 17, 18])

print(x_test.shape)
print(y_test.shape)


# 2. 모델 구성
# LSTM 은 층이 하나라도 많은 연산을 함
# return_sequences=True, 다음 계층 LSTM일때 LSTM 데이터로 넘겨주기 위한 옵션
model = Sequential()
model.add(LSTM(60, input_shape=(4, 1)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1))

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=0, callbacks=[early_stopping])


# 4. 평가
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss: ', loss)
print('acc: ', acc)
print('y_predict(x_test) : \n', y_predict)
