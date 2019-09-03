# 모듈 불러오기
import pyodbc as pyo

# 연결 문자열 셋팅
server = 'localhost'
database = 'bitdb'
username = 'bit'
password = '1234'

# 데이터 베이스 연결
cnxn = pyo.connect('DRIVER={ODBC Driver 13 for SQL Server}; SERVER=' + server +
                    '; PORT=1433; DATABASE=' + database +
                    '; UID=' + username +
                    '; PWD=' + password)

# 커서 생성
cursor = cnxn.cursor()

# 커서에 쿼리 입력 후 실행
tsql = 'SELECT * FROM iris2;'

with cursor.execute(tsql):
    # 한행 가져오기
    row = cursor.fetchone()
    # 행이 존재할 때까지, 하나씩 행을 증가시키면서 모든 컬럼을 공백으로 구분해 출력합니다.
    while row:
        print(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + ' ' +
            str(row[3]) + ' ' + str(row[4]))
        row = cursor.fetchone()

cnxn.close()