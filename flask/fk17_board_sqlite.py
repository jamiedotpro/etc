from flask import Flask, render_template, request
import sqlite3

import os
dir_path = os.path.dirname(os.path.abspath(__file__))
db_file = os.path.join(dir_path, 'wanggun.db')

app = Flask(__name__)

# 데이터베이스 확인
# conn = sqlite3.connect('.\\flask\\wanggun.db')
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute('SELECT * FROM general')
print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect(db_file)
    # conn.row_factory=sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM general')
    rows = c.fetchall();
    return render_template('board_index.html', rows=rows)

@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect(db_file)
    # conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id='+str(id))
    rows = c.fetchall();
    return render_template('board_modi.html', rows=rows)

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect(db_file) as con:
                cur = con.cursor()
                cur.execute('UPDATE general SET war=' + str(war) + ' WHERE id=' + str(id))
                con.commit()
                msg = '정상적으로 입력되었습니다.'
        except:
            con.rollback()
            msg = '입력과정에서 에러가 발생했습니다.'
        
        finally:
            return render_template('board_result.html', msg=msg)
            con.close()

app.run(host='127.0.0.1', port=5000, debug=False)