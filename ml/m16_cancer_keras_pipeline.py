from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 데이터 불러오기
cancer = load_breast_cancer()
# print(cancer.DESCR)

# df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
# sy = pd.Series(cancer.target, dtype="category")
# sy = sy.cat.rename_categories(cancer.target_names)
# df['class'] = sy
# print(df.tail())

# sns.pairplot(vars=["worst radius", "worst texture", "worst perimeter", "worst area"], 
#              hue="class", data=df)
# plt.show()

x = cancer.data
y = cancer.target
print(x.shape)
print(y.shape)


# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

'''
parameters = {
    'C': [1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': [0.001, 0.0001],
}

kfold_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(SVC(), parameters, cv=kfold_cv)
clf.fit(x_train, y_train)
print('최적의 매개 변수 =', clf.best_estimator_)


y_pred = clf.predict(x_test)
print('최종 정답률 = ', accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print('최종 정답률 = ', last_score)
'''


# 하이퍼 파라미터 최적화
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(30, ), name='input')
    x = Dense(60, activation='relu', name='hidden1')(inputs)
    x = Dense(90, activation='relu', name='hidden2')(x)
    x = Dense(50, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(20, activation='relu', name='hidden4')(x)
    prediction = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                    metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [5,10,20,30]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.0, 0.4, 5)
    return{'batch_size':batches, 'optimizer':optimizers, 'keep_prob':dropout}

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

# pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters,
                            n_iter=1, n_jobs=-1, cv=3, verbose=1)
                            # 작업이 10회 수행, 3겹 교차검증 사용

# pipe에서 search를 받아서 fit하면 내부의 search가 fit 된다.
pipe = make_pipeline(MinMaxScaler(), search)
pipe.fit(x_train, y_train)

print(search.best_params_)

print('score: ', search.score(x_test, y_test))
print('predict: \n', search.predict(x_test))


model_dic = search.best_params_

model = build_network(model_dic['keep_prob'], model_dic['optimizer'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, batch_size=model_dic['batch_size'], epochs=500, callbacks=[early_stopping])

print('acc: ', model.evaluate(x_test, y_test))
# acc:  [0.1325709614575955, 0.9473684168698495]