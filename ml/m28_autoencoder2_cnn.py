# 데이터-------------------------------------------
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)
print(x_train.shape)
print(x_test.shape)

# 모델 구성-------------------------------------------
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Model
from keras.regularizers import Regularizer

# 인코딩될 표현(representation)의 크기
encoding_dim = 32

# 입력 플레이스홀더
input_img = Input(shape=(28, 28, 1))
# 'encoded'는 입력의 인코딩된 표현
encoded = Conv2D(encoding_dim, (3, 3), activation='relu', padding='same')(input_img)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)

# 'decoded'는 입력의 손실있는 재구성 (lossy reconstruction)
decoded = Conv2D(encoding_dim, (3, 3), activation='relu', padding='same')(encoded)
# encoded = UpSampling2D((2, 2))(encoded)   # 이미지를 크기를 늘려준다
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)

# 입력을 입력의 재구성으로 매핑할 모델
autoencoder = Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
history = autoencoder.fit(x_train, x_train, epochs=100, batch_size=256,
                            shuffle=True, validation_data=(x_test, x_test),
                            callbacks=[early_stopping])

# 숫자들을 인코딩 / 디코딩
# test set에서 숫자들을 가져왔다는 것을 유의
decoded_imgs = autoencoder.predict(x_test)


# 이미지 출력-------------------------------------------
# Matplotlib 사용
import matplotlib.pyplot as plt

n = 10 # 몇 개의 숫자를 나타낼 것인지
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 그래프 출력-------------------------------------------
def plot_acc(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history
    
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)
    # plt.show()

def plot_loss(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)
    # plt.show()

plot_acc(history, '(a) 학습 경과에 따른 정확도 변화 추이')
plt.show()
plot_loss(history, '(b) 학습 경과에 따른 손실값 변화 추이')
plt.show()

loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)
