# 모듈을 불러오기
import pymssql as ms
import numpy as np

# 데이터 베이스에 연결
# conn = ms.connect(server='localhost', user='bit', password='1234', database='bitdb')
conn = ms.connect(server='127.0.0.1', user='bit', password='1234', database='bitdb')

# 커서 생성
cursor = conn.cursor()

# 커서에 쿼리 입력 후 실행
cursor.execute('SELECT * FROM iris2;')
# cursor.execute('SELECT TOP (10) * FROM train;')
row = cursor.fetchall()
# print(row)
conn.close()


aaa = np.asarray(row)
print(aaa)
print(aaa.shape)
print(type(aaa))

np.save('test_aaa.npy', aaa)
