import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization

# 1. 데이터

a = np.array(range(1, 101))
batch_size = 1
size = 5
def split_5(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)


dataset = split_5(a, size)
print('======================')
print(dataset)
print(dataset.shape)

x_train = dataset[:, 0:4]
y_train = dataset[:, 4]

x_train = np.reshape(x_train, (len(x_train), size-1, 1))


x_test = x_train + 100
y_test = y_train + 100

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_test[0])

# 2. 모델구성
model = Sequential()
# 상태유지 LSTM
    # 한번 fit 했을 때의 batch 값을 다음에 다시 사용하는 방식
    # 일반 LSTM보다 좀더 잘 맞는다고 함
    # batch_input_shape=(배치작업을 몇개씩 할 것인지, , )
    # stateful: 훈련했던 상태유지한다(true), 안한다(false, default)
# model.add(LSTM(128, batch_input_shape=(batch_size,4,1), stateful=True))
# # model.add(LSTM(128, return_sequences=True))
# # model.add(Dropout(0.5))
# # model.add(LSTM(128))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128))
# # model.add(BatchNormalization())
# model.add(Dense(128))
# model.add(Dense(128))
# # model.add(Dropout(0.5))
# model.add(Dense(128))
# model.add(Dense(1))
# mse :  3.4173729514562488
# [[111.82709 ]
#  [112.20453 ]
#  [113.03317 ]
#  [113.87483 ]
#  [114.694916]]
# RMSE :  34.93300980756885
# R2 :  -0.5891244808017821

model.add(LSTM(64, batch_input_shape=(batch_size,4,1), stateful=True))
model.add(Dense(128, activation='relu'))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0,
                        write_graph=True, write_images=True)
early_stoppng = EarlyStopping(monitor='loss', patience=20, mode='atuo')


num_epochs = 100

# 상태유지 LSTM은 fit 을 여러번함.
# 한번 fit이 끝나면
    # shuffle=False : 이전의 훈련했던 상태를 섞지 않겠다

for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=2,
                shuffle=False, validation_data=(x_test, y_test),
                callbacks=[early_stoppng, tb_hist])
    model.reset_states()
    # 훈련 상태가 변할지의 여부는 위에서 설정
    # 상태유지 LSTM은 fit할 때마다, evaluate 후에 reset_states() 호출해야함

mse, _ = model.evaluate(x_train, y_train, batch_size=batch_size)
print('mse : ', mse)
model.reset_states()

y_predict = model.predict(x_test, batch_size=batch_size)
print(y_predict[0:5])

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(ytest, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))   # 낮을수록 좋음

# R2 구하기. R2는 결정계수로 1에 가까울수록 좋음
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)

import matplotlib.pyplot as plt

# print(history.history.keys())
# dict_keys(['val_loss', 'val_mean_squared_error', 'loss', 'mean_squared_error'])
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

'''
x_train     y_train
1,2,3,4,    5
2,3,4,5,    6
...
96,97,98,99,    100

1. mse 값을 1이하로 만들기
        -> 3개 이상 히든레이어 추가 해서 할 것
        -> 드랍아웃 또는 BatchNormalization 적용
2. RMSE 함수 적용
3. R2 함수 적용

4. early stopping 기능 적용
5. tensorboard 적용

6. matplotlib 이미지 적용 mse/epochs
        -> 가로는 epo

'''