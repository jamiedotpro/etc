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

# flask 웹서버를 실행
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/sqltable')
def showsql():
    cursor.execute(tsql)
    return render_template('myweb.html', rows=cursor.fetchall())

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
