import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, KFold
import os

# 데이터 읽어 들이기
dir_path = os.getcwd()
winequality_white_file = os.path.join(dir_path, 'etc/data/winequality-white.csv')
wine = pd.read_csv(winequality_white_file, sep=';', encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop('quality', axis=1)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

parameters = {
    'n_estimators': [100, 200, 500],
    'max_depth': [2, 3, 4],
    'n_jobs': [-1]
}

kfold_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold_cv)
clf.fit(x_train, y_train)
print('최적의 매개 변수 =', clf.best_estimator_)

y_pred = clf.predict(x_test)
print('최종 정답률 = ', accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print('최종 정답률 = ', last_score)

# 최적의 매개 변수 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
#        max_delta_step=0, max_depth=4, min_child_weight=1, missing=None,
#        n_estimators=500, n_jobs=-1, nthread=None,
#        objective='multi:softprob', random_state=0, reg_alpha=0,
#        reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
#        subsample=1, verbosity=1)
# 최종 정답률 =  0.939795918367347
# 최종 정답률 =  0.939795918367347