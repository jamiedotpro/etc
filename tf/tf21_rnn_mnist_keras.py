from keras.datasets import mnist
from keras.utils import np_utils
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], 28, 28).astype('float32') / 255
print(train_labels.shape)
print(test_labels.shape)
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

from keras import models, layers
from keras.layers import Dense, LSTM

model = models.Sequential()
model.add(LSTM(64, activation='relu', input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(train_images, train_labels, epochs=15, batch_size=128, callbacks=[early_stopping])

loss, acc = model.evaluate(test_images, test_labels)

print('acc:', acc)