# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers
model = Sequential()

# 깊이, 노드 개수, epochs, batch_size 순으로 조정
model.add(Dense(50, input_shape=(3, ), activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# model.summary()

model.save('savetest01.h5')
print('저장 완료')