from flask import Flask

app = Flask(__name__)

# app.route는 바로 밑에 함수와 연결된다. 함수명은 아무거나 상관없음
@app.route('/')
def hello():
    return '<h1>hello world</h1>'

@app.route('/bit/')
def hello2():
    return '<h1>hello bit computer world</h1>'

if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=5000, debug=False)
    app.run(host='192.168.0.119', port=8888, debug=False)
