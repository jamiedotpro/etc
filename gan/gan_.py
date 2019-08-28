import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model
import os


class Generator(object):
    # 변수 초기화
    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        self.w = width
        self.h = height
        self.c = channels
        self.optimizer = Adam(lr=0.0002, decay=8e-9)

        self.latent_space_size = latent_size
        self.latent_space = np.random.normal(0, 1, (self.latent_space_size,))

        self.generator = self.model()
        self.generator.compile(loss='binary_crossentropy',
                                optimizer=self.optimizer)

        self.generator.summary()


    # 생성기 모델을 구축해 반환
    def model(self, block_starting_size=128, num_blocks=4):
        # 먼저 모델을 정의하고, 기본 Sequential 구조로 시작
        model = Sequential()

        # 다음으로, 우리는 신경망에서 첫 번째 계층 블록을 시작
        block_size = block_starting_size
        model.add(Dense(block_size, input_shape=(self.latent_space_size,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 이 블록은 잠재 표본의 입력 모양이자 초기 블록 크기의 시작 크기에 맞춰 신경망에 조밀(dense) 계층을 추가한다.
        # 이 경우에는 128개의 뉴런으로 시작한다. LeakyReLU 활성 계층을 사용하여,
        # 우리는 경사 소멸(vanishing gradients)과 비활성 뉴런(non-activated neurons)을 피할 수 있다.
        # 그런 다음 BatchNormalization은 이전 계층을 기반으로 활성을 정규화해 계층을 정리한다.
        # 이렇게 하면 신경망의 효율성이 향상된다.

        for i in range(num_blocks-1):
            block_size = block_size * 2
            model.add(Dense(block_size))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            # 이 코드 집합은 이전 블록과 같은 추가 블록을 추가할 수 있지만 조밀 계층 크기를 두 배로 만든다.
            # 블록 수를 다르게 하여 시험해보기. 어떤 결과가 나올지.. 성능이 향상되는가? 더 빠르게 수렴하는가? 아니면 발산하는가?
            # 이 코드 집합은 더 유연한 방식으로 이런 아키텍처 유형을 실험해 볼 수 있게 해 줘야 한다.

        # 이 메서드의 마지막 부분은 출력을 입력 이미지와 동일한 모양으로 재구성해 모델을 반환한다.
        model.add(Dense(self.w * self.h * self.c, activation='tanh'))
        model.add(Reshape((self.w, self.h, self.c)))

        return model

    # 모델을 요약한 내용을 화면에 인쇄
    def summary(self):
        # 케라스에서 제공하는 텍스트 요약
        return self.generator.summary()

    # 모델 구조를 데이터 폴더의 파일로 저장
    def save_model(self):
        plot_model(self.generator.model, to_file='generator_model.png')



class Discriminator(object):
    # 변수 초기화
    # 너비, 높이, 채널 및 잠재 공간 크기가 있는 클래스를 초기화한다
    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        self.capacity = width*height*channels
        self.shape = (width, height, channels)
        self.optimizer = Adam(lr=0.0002, decay=8e-9)

        # 모델 초기화
        self.discriminator = self.model()

        # binary_crossentropy 손실과 지정된 최적화기를 사용해 모델을 컴파일한다
        self.discriminator.compile(loss='binary_crossentropy',
                                    optimizer=self.optimizer, metrics=['accuracy'])

        # 모델의 텍스트 요약을 표시한다
        self.discriminator.summary()
    
    # 이진 분류기를 만들어 반환
    def model(self):
        # 이 메서드는 순차적 모델로 시작된다. 이렇게 하면 계층들을 서로 쉽게 접합할 수 있다.
        # 케라스는 처리 도중에 몇 가지 가정을 하는데, 예를 들면 이전 계층의 크기가 처리 진행 중인 계층의 입력과 같다는 식이다.
        model = Sequential()

        # 첫 번째 계층
        # 이 계층은 데이터를 단일 데이터 스트림으로 전개한다.
        model.add(Flatten(input_shape=self.shape))

        # 다음 계층은 처리해야 할 작업 중 가장 앞서 부분을 수행할 것이다.
        model.add(Dense(self.capacity, input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        # 조밀 계층(dense layer)이란 간단히 말하면 각 뉴런들이 이전 계층의 뉴런들과 서로 모두 완전히 연결된 계층을 말한다.
        # 이런 계층은 신경망의 기본 빌딩 블록 중 하나이며 입력이 각 뉴런에 도달할 수 있게 한다.
        # LeakyReLU는 신경망을 구성하는 유닛이 작동하지 않을 때 작은 경사를 사용할 수 있도록 하는 특별한 활성 계층 유형이다.
        # 실제로 LeakyReLU를 활성 함수로 쓰는 경우에 활성치가 0에 가까워져도 비활성 유닛을 처리할 수 있으므로 일반적인 ReLU보다 유리하다.

        # 다음 블록은 단순히 이 계층에서 사용할 수 있는 용량을 절반으로 줄여서 신경망을 거치면서
        # 중요한 특징들을 학습할 수 있게 해준다.
        model.add(Dense(int(self.capacity/2)))
        model.add(LeakyReLU(alpha=0.2))

        # 마지막으로 입력이 계급의 일부일 확률이나 일부가 아닐 확률을 나타내는 최종 계층이 있다.
        model.add(Dense(1, activation='sigmoid'))

        return model

    # 모델을 요약한 내용을 화면에 인쇄
    def summary(self):
        return self.discriminator.summary()

    # 모델 구조를 데이터 폴더의 파일로 저장
    def save_model(self):
        plot_model(self.discriminator.model, to_file='discriminator_model.png')



class GAN(object):
    # 변수 초기화
    def __init__(self, discriminator, generator):
        self.optimizer = Adam(lr=0.0002, decay=8e-9)
        self.generator = generator

        self.discriminator = discriminator
        self.discriminator.trainable = False
        # 판별기의 훈련 가능성을 False로 설정. 이는 적대 훈련을 하는 중에는 판별기가 훈련이 되지 않게 하겠다는 것
        # 이에 따라 생성기는 지속적으로 개선되지만 판별기는 원래대로 유지된다.
        
        self.gan_model = self.model()
        self.gan_model.compile(loss='binary_crossentropy',
                                optimizer=self.optimizer)
        
        self.gan_model.summary()
    
    # 적대 모델을 구축하고 반환
    def model(self):
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    # 모델을 요약한 내용을 화면에 인쇄
    def summary(self):
        return self.gan_model.summary()

    # 모델 구조를 데이터 폴더의 파일로 저장
    def save_model(self):
        plot_model(self.gan_model.model, to_file='gan_model.png')

from random import randint
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

class Trainer():
    def __init__(self, width=28, height=28, channels=1, latent_size=100,
                    epochs=50000, batch=32, checkpoint=50, model_type=-1):
        self.w = width
        self.h = height
        self.c = channels
        self.epochs = epochs
        self.batch = batch
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.latent_space_size = latent_size

        # Generator, Discriminator 클래스 초기화
        self.generator = Generator(height=self.h, width=self.w, channels=self.c,
                                    latent_size=self.latent_space_size)
        self.discriminator = Discriminator(height=self.h, width=self.w, channels=self.c)
        self.gan = GAN(generator=self.generator.generator, discriminator=self.discriminator.discriminator)

        self.load_mnist()

    def load_mnist(self, model_type=3):
        allowed_types = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if self.model_type not in allowed_types:
            print('ERROR: Only Integer Values from -1 to 9 are allowed')

        (self.x_train, self.y_train), (_, _) = mnist.load_data()
        if self.model_type != -1:
            self.x_train = self.x_train[np.where(self.y_train==int(self.model_type))[0]]
        self.x_train = (np.float32(self.x_train) - 127.5) / 127.5
        self.x_train = np.expand_dims(self.x_train, axis=3)
        return

    # numpy 호출을 사용하기 쉬운 메서드 호출로 대체
    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances, self.latent_space_size))

    # 모델 검사점을 그려내는 코드
    # 이 함수는 생성기 출력의 무작위 표본을 보여주는 그림을 표시한다.
    # 다음과 같이 검사점 이미지를 그리는 메서드를 정의. 이때 숫자값 e를 입력으로 사용한다.
    def plot_checkpoint(self, e):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(curr_dir, 'data/sample_' + str(e) + '.png')
        # filename = 'sample_' + str(e) + '.png'

        # 잠재 공간에서 잡음을 생성하고 생성기로 이미지를 생성하는 코드는 다음과 같다.
        noise = self.sample_latent_space(16)
        images = self.generator.generator.predict(noise)

        # 이렇게 새로 생성된 이미지들을 그리는 코드는 다음과 같다
        # 이 경우에 각 에포크 검사점마다 16개 이미지가 생성된다.
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.h, self.w])
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            # 마지막으로 그림을 그려내고 저장한 다음에 그림을 닫는 코드는 다음과 같다
            plt.tight_layout()
            plt.savefig(filename)
            plt.close('all')
            return

    def train(self):
        for e in range(self.epochs):
            # 훈련 데이터셋에서 무작위 이미지들로 구성된 배치 한 개를 가져와서 x_real_images와 y_real_labels 변수를 만들 것이다.

            # 배치를 부여잡는다
            count_real_images = int(self.batch/2)
            starting_index = randint(0, (len(self.x_train) - count_real_images))
            real_images_raw = self.x_train[starting_index : (starting_index + count_real_images)]
            x_real_images = real_images_raw.reshape(count_real_images, self.w, self.h, self.c)
            y_real_labels = np.ones([count_real_images, 1])

            # batch 변수를 사용해서 지정한 이미지의 수를 반으로 줄였는데,
            # 그 이유는 다음 단계에서 생성기로 이미지를 생성해 배치의 나머지 반을 채울 것이기 때문이다.
            # 이 훈련 배치용으로 생성된 이미지를 부여잡는다.
            latent_space_samples = self.sample_latent_space(count_real_images)
            x_generated_images = self.generator.generator.predict(latent_space_samples)
            y_generated_labels = np.zeros([self.batch - count_real_images, 1])

            # 이제 훈련용으로 쓸 전체 배치를 개발함. 이 두 집합을 x_batch, y_batch 변수에 연결해 훈련해야 한다

            # 판별기에서 훈련용으로 결합
            x_batch = np.concatenate([x_real_images, x_generated_images])
            y_batch = np.concatenate([y_real_labels, y_generated_labels])
            # 흥미로운 부분은 우리는 이 배치를 사용해 판별기를 훈련하고 있다.
            # 판별기는 훈련될 때 이미지들이 진짜가 아니라는 것을 알고 있으므로,
            # 판별기는 생성된 이미지와 진짜 이미지 사이에서 끊임 없이 결함(imperfections)을 찾으려고 할 것이다.

            # 판별기를 훈련하고 손실값을 파악해 알릴 수 있게 하자
            # 이제 다음과 같은 배치를 사용해 판별기를 훈련한다.
            discriminator_loss = self.discriminator.discriminator.train_on_batch(x_batch, y_batch)[0]
            # 우리는 이제 생성기가 출력해 낸 이미지(따라서 적절치 않은 레이블이 붙은 이미지)를 사용해 GAN을 훈련할 것이다.
            # 즉 우리는 잡음을 바탕으로 삼아 이미지를 생성해 내고, GAN을 훈련할 때 그러한 이미지 중 하나에 레이블을 지정한다.
            # 이유는 이게 새로 훈련된 판별기를 사용해 생성된 출력을 개선하는 훈련 중에 소위 적대 훈련 부분(adversarial training portion)이기 때문이다.
            # GAN 손실 보고서에는 생성된 출력으로 인해 판별기가 혼란스러워 하는 면이 나타나게 된다.

            # 생성기를 훈련하는 코드는 다음과 같다
            # 잡음 생성
            x_latent_space_samples = self.sample_latent_space(self.batch)
            y_generated_labels = np.ones([self.batch, 1])
            generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

            # 손실 계량기준을 화면에 표시하고 데이터 폴더에 출력된 이미지로 모델을 확인하는 두 가지 부분이 스크립트 끝 부분에 들어가야 한다.
            print('Epoch: ' + str(int(e)) + ', [Discriminator :: Loss: ' + str(discriminator_loss) + '], ' +
                                                '[Generator :: Loss: ' + str(generator_loss) + ']')
            if e % self.checkpoint == 0:
                self.plot_checkpoint(e)
            return


if __name__ == '__main__':
    height = 28
    width = 28
    channel = 1
    latent_space_size = 100
    epochs = 50001
    batch = 32
    checkpoint = 500
    model_type = -1
    trainer = Trainer(height=height, width=width, channels=channel,
                        latent_size=latent_space_size, epochs=epochs,
                        batch=batch, checkpoint=checkpoint, model_type=model_type)
    trainer.train()