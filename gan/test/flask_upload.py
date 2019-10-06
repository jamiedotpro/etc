from flask import Flask, render_template, request
from werkzeug import secure_filename
import flask
import socket
import time
HOST ='104.154.169.173'
PORT = 9009
PORT2 = 9010


app = Flask(__name__)
def getFileFromServer(filename):
    data_transferred = 0
 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST,PORT))
        sock.sendall(filename.encode())
 
        data = sock.recv(1024)
        if not data:
            print('파일[%s]: 서버에 존재하지 않거나 전송중 오류발생' %filename)
            return
 
        with open(filename,'wb') as f:
            try:
                while  data:
                    f.write(data)
                    data_transferred += len(data)
                    data = sock.recv(1024)
            except Exception as e:
                print(e)


#업로드 HTML 렌더링
@app.route('/upload')
def render_file():
   return render_template('upload.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      
      f = flask.request.files['file']
      #저장할 경로 + 파일명
      f.save(secure_filename(f.filename))
      
      getFileFromServer("1.png")
      return 'uploads 디렉토리 -> 파일 업로드 성공!'

if __name__ == '__main__':
    #서버 실행
   app.run(host = '0.0.0.0', port=5000, threaded= True  ,debug=False)
