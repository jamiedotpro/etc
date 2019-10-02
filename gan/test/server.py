import socketserver, threading
from os.path import exists
import socket

HOST =''
PORT = 9009 # 다운로드 서버 -> 클라이언트
PORT2 = 9010 # 업로드 클라이언트 -> 서버
ADDR1=((HOST, PORT))
ADDR2=((HOST, PORT2))

def client(ip, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        sock.send(message.encode('utf-8'))
        response = sock.recv(1024).decode()
        print("Received: {}".format(response))
    finally:
        sock.close()

class MyTcpHandler(socketserver.StreamRequestHandler):
    
    
    timeout = 3
    def handle(self):
        self.request.settimeout(1)
        
        data_transferred = 0
        print('[%s] 연결됨' %self.client_address[0])
        filename = self.request.recv(1024) # 클라이언트로 부터 파일이름을 전달받음
        filename = filename.decode() # 파일이름 이진 바이트 스트림 데이터를 일반 문자열로 변환
        cur_thread = threading.current_thread()
        if not exists(filename): # 파일이 해당 디렉터리에 존재하지 않으면
            return # handle()함수를 빠져 나온다.
 
        print('파일[%s] 전송 시작...' %filename)
        with open(filename,'rb') as f:
            try:
                data = f.read(1024) # 파일을 1024바이트 읽음
                while data: # 파일이 빈 문자열일때까지 반복
                    data_transferred += self.request.send(data)
                    data = f.read(1024)
            except Exception as e:
                print(e)
 
        print('전송완료[%s], 전송량[%d]' %(filename,data_transferred))

class MyTcpUpload(socketserver.BaseRequestHandler):
    timeout = 3
    # self.request.settimeout(1)
    def handle(self):
        
        # self.request.settimeout(1)

        data_transferred = 0
        print('[%s] 연결됨' %self.client_address[0])
        sock = self.request

        filename = self.request.recv(1024) # 클라이언트로 부터 파일이름을 전달받음
        filename = filename.decode()
        print(filename)
        # if exists(filename): # 파일이 해당 디렉터리에 존재하면
        #     self.request.send("1010")
        #     return # handle()함수를 빠져 나온다.
        # print(1)
        data = self.request.recv(1024) # 클라이언트로 부터 이미지를 전달받음
        if not data:
            print('파일[%s]: 서버에 존재하지 않거나 전송중 오류발생' %filename)
            return

        with open('upload/' + filename,'wb') as f:
            try:
                while  data:
                    f.write(data)
                    data_transferred += len(data)
                    data = sock.recv(1024)
                    
            except Exception as e:
                print(e)
        print('파일[%s] 전송종료. 전송량 [%d]' %(filename, data_transferred))



import select
def runServer():
    print('++++++파일 서버를 시작++++++')
    print("+++파일 서버를 끝내려면 'Ctrl + C'를 누르세요.")
    server1 = socketserver.TCPServer(ADDR1,MyTcpHandler)
    server2 = socketserver.TCPServer(ADDR2,MyTcpUpload)
    ip1, port1 = server1.server_address
    ip2, port2 = server2.server_address
    read_port_list =[server1,server2]
    
    while True:
        print("wait~~~~~~")
        conn_read_socket_list, conn_write_socket_list, conn_except_socket_list = select.select(read_port_list, [], [])
        # print(conn_read_socket_list)
        for conn_read_socket in conn_read_socket_list:
            if conn_read_socket == server1:
                print("MODE DOWNLOAD")
                server_thread = threading.Thread(target=server1.serve_forever)
                server_thread.daemon = True
                server_thread.start()
            
            elif conn_read_socket == server2:
                print("MODE UPLOAD")
                # server2.handle_request()
                server_thread = threading.Thread(target=server2.serve_forever)
                server_thread.daemon = True
                server_thread.start()
        
        # try:
        #     print('MODE : DOWNLOAD')
            
            
        #     # server.handle_request()
            
        # except KeyboardInterrupt:
        #     print('++++++파일 서버를 종료합니다.++++++')

        # try:
        #     print('MODE : UPLOAD')
            
            

        #     # server.serve_forever()
        
        # except KeyboardInterrupt:
        #     print('++++++파일 서버를 종료합니다.++++++')
        
 
 
runServer()