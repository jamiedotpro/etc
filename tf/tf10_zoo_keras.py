from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np
import os

dir_path = os.getcwd()
zoo_file = os.path.join(dir_path, 'etc/data/data-04-zoo.csv')
xy = np.loadtxt(zoo_file, delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

y_data = np_utils.to_categorical(y_data)

model = Sequential()
model.add(Dense(7, activation='softmax', input_shape=(x_data.shape[1],)))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(x_data, y_data, batch_size=1,
                    epochs=2000, validation_split=0.2, verbose=1,
                    callbacks=[early_stopping_callback])

score = model.evaluate(x_data, y_data)
print('\ntest score:', score[0])
print('test accuracy:', score[1])
