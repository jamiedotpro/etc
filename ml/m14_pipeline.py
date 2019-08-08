import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
import os

# 붓꽃 데이터 읽어 들이기
dir_path = os.getcwd()
iris_data_file = os.path.join(dir_path, 'etc/data/iris2.csv')
iris_data = pd.read_csv(iris_data_file, encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = iris_data.loc[:, 'Name']

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
# from sklearn.pipeline import make_pipeline
# make_pipeline과 pipeline의 차이는 디폴트로 지정된 키 이름을 사용할지, 키 이름을 직접 입력할지의 차이임
# pipe = make_pipeline(MinMaxScaler(), SVC((C=10)) # == pipe = Pipeline([('minmaxscaler', MinMaxScaler()), ('svm', SVC())])
pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])

pipe.fit(x_train, y_train)

print('테스트 점수: ', pipe.score(x_test, y_test))
