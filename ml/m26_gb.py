import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
}

kfold_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(GradientBoostingRegressor(), parameters, cv=kfold_cv)
clf.fit(x_train, y_train)
print('최적의 매개 변수 =', clf.best_estimator_)

y_pred = clf.predict(x_test)
last_score = clf.score(x_test, y_test)
print('최종 정답률 = ', last_score)

# 최적의 매개 변수 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#              learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
#              max_leaf_nodes=None, min_impurity_decrease=0.0,
#              min_impurity_split=None, min_samples_leaf=1,
#              min_samples_split=2, min_weight_fraction_leaf=0.0,
#              n_estimators=100, n_iter_no_change=None, presort='auto',
#              random_state=None, subsample=1.0, tol=0.0001,
#              validation_fraction=0.1, verbose=0, warm_start=False)
# 최종 정답률 =  0.9174968888808568