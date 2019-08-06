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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 학습하기
# n_estimators: 생성할 트리 개수
# max_features: 최대 선택할 특성의 수
# n_jobs : int 또는 None, 선택 사항 (기본값 = 없음). fit and predict에 대해 동시에 실행할 작업 수.
#         -1 means using all processors
# n_estimators에서
# bootstrap sampling은 random forest의 tree가 조금씩 다른 데이터셋으로 만들어지도록합니다.
# max_feature에서 각 node는 특성의 일부를 무작위로 추출하여 사용합니다.
# max_features를 전체 특성의수로 설정하면 모든 특성을 고려하므로 decision tree에서 무작위성이 들어가지 않습니다.
# 그러나 bootstrap sampling의 무작위성은 존재합니다.
# max_features 값을 크게 하면 
# random forest의 tree들은 같은 특성을 고려하므로 tree들이 매우 비슷해지고 가장 두드러진 특성을 이용해 데이터에 잘 맞춰집니다.
# max_features를 낮추면 
# random forest tree들은 많이 달라지고 각 tree는 데이터에 맞추기 위해 tree의 깊이가 깊어집니다.
model = RandomForestClassifier(n_estimators=100, max_features=7, oob_score=True, n_jobs = -1, random_state=42)
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print('정답률=', accuracy_score(y_test, y_pred))
print(aaa)