import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
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

# 학습하기
# n_estimators
# max_features
# max_depth
# criterion
# model = RandomForestClassifier(n_estimators=100, max_features=7, oob_score=True, n_jobs = -1, random_state=42)
# 정답률= 0.9408163265306122
# model = RandomForestClassifier(n_estimators=1000, max_features=5, oob_score=True, n_jobs = -1, random_state=42)
# 정답률= 0.9438775510204082
# model = RandomForestClassifier(n_estimators=1000, max_features=9, oob_score=True, n_jobs = -1, random_state=42)
# 정답률= 0.9448979591836735
# model = RandomForestClassifier(n_estimators=1000, max_features=10, oob_score=True, n_jobs = -1, random_state=42)
# 정답률= 0.9469387755102041
# model = RandomForestClassifier(n_estimators=200, max_features=6, max_depth=20, oob_score=True, n_jobs = -1, random_state=42)
# 정답률= 0.9530612244897959
model = RandomForestClassifier(n_estimators=200, max_features=3, max_depth=20, oob_score=True, n_jobs = -1, random_state=42)
# 정답률= 0.9561224489795919


model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)   # R2 score

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print('정답률=', accuracy_score(y_test, y_pred))
print(aaa)