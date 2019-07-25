# 1. 데이터
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# x2 = np.array([4, 5, 6])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()    # 순차적으로 내려가는 모델이 Sequential

model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

model.summary()
# param의 수 : 노드가 5개일 때. 바이어스는 노드 수만큼 차지함
# 입력값(1) * 첫 번째 레이어 뉴런(5) + 첫 번째 레이어의 뉴런에 대한 바이어스(5)
# 입력값(5) * 두 번째 레이어 뉴런(3) + 두 번째 레이어의 뉴런에 대한 바이어스(3)
# 입력값(3) * 세 번째 레이어 뉴런(4) + 세 번째 레이어의 뉴런에 대한 바이어스(4)
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_1 (Dense)              (None, 5)                 10
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 18
# _________________________________________________________________
# dense_3 (Dense)              (None, 4)                 16
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 5
# =================================================================
# Total params: 49
# Trainable params: 49
# Non-trainable params: 0

'''
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x, y, epochs=200, batch_size=3)
model.fit(x, y, epochs=100)

# 4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=3)
print('acc : ', acc)

y_predict = model.predict(x2)
print(y_predict)
'''