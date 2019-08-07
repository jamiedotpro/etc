import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import warnings
from sklearn.utils.testing import all_estimators
import os
import numpy as np

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
dir_path = os.getcwd()
iris_data_file = os.path.join(dir_path, 'etc/data/iris2.csv')
iris_data = pd.read_csv(iris_data_file, encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = iris_data.loc[:, 'Name']

# classifier 알고리즘 모두 추출하기
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

# K-분할 크로스 밸리데이션 전용 객체
# 5조각으로 나눠서 4개를 트레인셋, 1개를 테스트셋
for spl_cnt in range(3, 11):
    kfold_cv = KFold(n_splits=spl_cnt, shuffle=True)

    # cnt = 0
    score_list = []
    for(name, algorithm) in allAlgorithms:
        # 각 알고리즘 객체 생성하기
        clf = algorithm()

        # score 메서드를 가진 클래스를 대상으로 하기
        if hasattr(clf, 'score'):
            scores = cross_val_score(clf, x, y, cv=kfold_cv)    # 교차검증
            # print(name, '의 정답률= ', end='')
            # print(scores)
            score_list.append([name, np.mean(scores)])
            # cnt += 1

    print('-------------------------------------------------------------------------------------------------------')
    print('n_splits=', spl_cnt, '일 때: ')
    # print('score_list 전체: \n', score_list)
    print('score_list 최대값 상위 3개: \n', sorted(score_list, key=lambda x: x[1], reverse=True)[:3])
    # print('개수: ', cnt)
