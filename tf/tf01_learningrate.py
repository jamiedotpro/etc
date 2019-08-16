# 1. 데이터
import numpy as np
x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))


# 3. 훈련
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
# optimizer = Adam(lr=0.01)  # lr=learningrate
# optimizer = SGD(lr=0.01)
# optimizer = RMSprop(lr=0.01)
# optimizer = Adagrad(lr=0.009)
# optimizer = Adadelta(lr=0.12)
# optimizer = Adamax(lr=0.02)
optimizer = Nadam(lr=0.01)
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가 예측
mse, _ = model.evaluate(x, y, batch_size=1)
print('mse : ', mse)
pred1 = model.predict([1.5, 2.5, 3.5])
print(pred1)
