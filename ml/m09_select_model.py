import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
import os

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
dir_path = os.getcwd()
iris_data_file = os.path.join(dir_path, 'etc/data/iris2.csv')
iris_data = pd.read_csv(iris_data_file, encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = iris_data.loc[:, 'Name']

# 학습 전용과 테스트 전용 분리하기
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# classifier 알고리즘 모두 추출하기--- (*1)
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')   # 나중에 label encoder 추가해서 확인하기

# all_estimators:
# 모든 사이킷런의 모델이 들어가 있다
#print(allAlgorithms)    # 모델 리스트 확인
print(len(allAlgorithms))   # scikit-learn 버전에 따라 다름. scikit-learn(0.20.3)은 31개
print(type(allAlgorithms))


# 하나만 지울 때 사용 가능
# allAlgorithms.remove(allAlgorithms[0])

# 일부만 가져와서 테스트 중...
# allAlgorithms = allAlgorithms[40:]


'''
# 리스트 내 중복된 값이 있을 때 전부 삭제하는 방법
del_algorithm = allAlgorithms[:1]   # 내부는 튜플이어서, 리스트 배열 일부만 가져와서 처리
new_algorithms = [x for x in allAlgorithms if x not in del_algorithm]
# c = [x for x in a if x not in b]

print(new_algorithms)
print(len(new_algorithms))
print(type(new_algorithms))
'''


# 모든 모델에 대해 학습하고 평가해서 정답률 출력
for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기--- (*2)
    clf = algorithm()

    # 학습하고 평가하기--- (*3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name, '의 정답률=', accuracy_score(y_test, y_pred))   # classifier
    # print(name, '의 정답률=', clf.score(y_test, y_pred)         # regressor
