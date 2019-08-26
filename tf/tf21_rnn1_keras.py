import tensorflow as tf
import numpy as np

idx2char = ['e', 'h', 'i', 'l', 'o']
_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1, 1)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype('float32')    # 사이킷런은 자동으로 알파벳 순서로 OneHotEncoder 처리해줌. float64로 리턴되기에 float32 로 형변환함

x_data = _data[:6,] # (6, 5)
y_data = _data[1:,] # (6, 5)
# y_data = np.argmax(y_data, axis=1)

x_data = x_data.reshape(1, 6, 5)    # (1, 6, 5)
y_data = y_data.reshape(1, 6, 5)

num_classes = 5
batch_size = 1          # (전체행)
sequence_length = 6     # 컬럼
input_dim = 5           # 몇개씩 작업
hidden_size = 5         # 첫번째 노드 출력 개수
learning_rate = 0.1

from keras import models, layers
from keras.layers import Dense, LSTM

model = models.Sequential()
model.add(LSTM(30, input_shape=(sequence_length, input_dim), return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(5, activation='softmax', return_sequences=True))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_data, y_data, epochs=700, batch_size=128, callbacks=[early_stopping])

loss, acc = model.evaluate(x_data, y_data)
pred = model.predict(x_data)

print('acc:', acc)
print(pred)

pred = np.argmax(pred, axis=2)
result_str = [idx2char[c] for c in np.squeeze(pred)]
print('\nPrediction str: ', ''.join(result_str))