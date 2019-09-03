# 모듈을 불러오기
import pymssql as ms
import numpy as np

# 데이터 베이스에 연결
# conn = ms.connect(server='localhost', user='bit', password='1234', database='bitdb')
conn = ms.connect(server='127.0.0.1', user='bit', password='1234', database='bitdb')

# 커서 생성
cursor = conn.cursor()

# 커서에 쿼리 입력 후 실행
# cursor.execute('SELECT * FROM iris2;')
cursor.execute('SELECT TOP (10) * FROM train;')

# 한 행 가져오기
row = cursor.fetchone()
print(type(row))  # tuple

# 행이 존재할 때까지, 하나씩 행을 증가시키면서...
while row:
    # print('첫 컬럼%s, 둘 컬럼=%s' % (row[0], row[1])) # 1-2번째 컬럼 출력
    print(row)  # 행 데이터 전체 출력
    row = cursor.fetchone()

# 연결 닫기
conn.close()