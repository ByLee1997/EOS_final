import socket
import cv2
import numpy
from matplotlib import pyplot as plt

def recvall(conn, count):
    buf = b''
    while count:
        newbuf = conn.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

TCP_IP = "192.168.0.38"
TCP_PORT = 8787
my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
my_socket.bind((TCP_IP, TCP_PORT))
my_socket.listen(True)
conn, addr = my_socket.accept()
while 1:
    stringData = recvall(conn, 320*240)
    # stringData = recvall(conn, int(length))
    if stringData:
        # print(len(stringData))
        data = numpy.fromstring(stringData, dtype='uint8')
        data = data.reshape((240, 320))
        # decimg=cv2.imdecode(data,1)
        cv2.imshow('SERVER', data)
        cv2.waitKey(30)

my_socket.close()
cv2.destroyAllWindows()