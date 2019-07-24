# 1. 데이터
import numpy as np
# x_train = x는 1부터 100
# y_train = y는 501부터 600

# x_test = 1001~1100
# y_test = 1101~1200

x_train = []
y_train = []
x_test = []
y_test = []


for i in range(1, 101):
    x_train.append(i)
    y_train.append(500 + i)
    x_test.append(1000 + i)
    y_test.append(1100 + i)
#np.arrange
#np.array([i for i in range(1, 101)])

# 범위를 맞추기 위해 정규화
def normalize(list):
    list_min = min(list)
    list_max = max(list)
    new_list = []
    for li in list:
        n = ((li - list_min) / (list_max - list_min))
        new_list.append(n)
    return new_list

def denormalize(list, mmax, mmin):
    new_list = []
    for li in list:
        n = round(li[0] * (mmax-mmin) + mmin)
        new_list.append(n)
    return new_list

xn_train = normalize(x_train)
yn_train = normalize(y_train)
xn_test = normalize(x_test)
yn_test = normalize(y_test)
print(yn_test)

# print('yn_train, normalize: ', yn_train)
# new = denormalize(yn_train, max(y_train), min(y_train))
# print('yn_train, denormalize: ', new)


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(xn_train, yn_train, epochs=200, batch_size=1)
model.fit(x_train, y_train, epochs=200, batch_size=1)

model.summary()

# 4. 평가 예측
#loss, acc = model.evaluate(xn_test, yn_test, batch_size=1)
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', acc)

y_predict = model.predict(x_test)
print(y_predict)

#y_predict = model.predict(xn_test)
#print(y_predict)
#re_y_predict = denormalize(y_predict, max(y_test), min(y_test))
#print('ytest: ', y_test)
#print('re_y_predict: ', re_y_predict)