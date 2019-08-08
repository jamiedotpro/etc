import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, KFold
import os

# 기온 데이터 읽어 들이기
dir_path = os.getcwd()
tem10y_file = os.path.join(dir_path, 'etc/data/tem10y.csv')
df = pd.read_csv(tem10y_file, encoding='utf-8')

# 데이터를 학습 전용과 테스트 전용으로 분리하기
train_year = (df['연'] <= 2015)
test_year = (df['연'] >= 2016)
interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = []  # 학습 데이터
    y = []  # 결과
    temps = list(data['기온'])
    for i in range(len(temps)):
        if i < interval:
            continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

x_train, y_train = make_data(df[train_year])
x_test, y_test = make_data(df[test_year])

parameters = {
    'n_estimators': [100, 200, 500],
    'max_depth': [2, 3, 4],
    'n_jobs': [-1]
}

kfold_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold_cv)
clf.fit(x_train, y_train)
print('최적의 매개 변수 =', clf.best_estimator_)

y_pred = clf.predict(x_test)
last_score = clf.score(x_test, y_test)
print('최종 정답률 = ', last_score)
