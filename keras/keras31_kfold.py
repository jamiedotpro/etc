from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

# ...Scaler 사용해서 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()

scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

from keras import models
from keras import layers

def build_model():
    # 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다.
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

seed = 77
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
model = KerasRegressor(build_fn=build_model, epochs=10,
                        batch_size=1, verbose=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(model, train_data, train_targets, cv=kfold)   # cv == cross Validation 교차검증

import numpy as np
print(results)
print(np.mean(results))

# 1. 사이킷런의 KFold로 리파인 시킬 것
# 2. 정규화 표준화 시킬 것
# 3. np.mean(all_scores)를 1 이하로 낮출 것


# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5)
# for train_index, test_index in kf.split(train_data, train_targets):
#     partial_train_data, val_data = train_data[train_index], train_data[test_index]
#     partial_train_targets, val_targets = train_targets[train_index], train_targets[test_index]
#     print(train_index, test_index)

