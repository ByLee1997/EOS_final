#!/usr/bin/python

import sys
import os
import time
import socket
import signal
import matplotlib.pyplot as plt
import numpy as np
import pickle
import struct
from PIL import Image
import random
from torchvision import transforms
import cv2

def start_tcp_client(server_ip, server_port):
    #create socket
    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        tcp_client.connect((server_ip, server_port))
    except socket.error as error_msg:
        print('fail to setup socket connection %s'%error_msg)
    return tcp_client

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    #print(sys.argv[0],sys.argv[1],sys.argv[2])
    path= 'school_dag/train/'+sys.argv[3]
    extensions= ['JPG','PNG']
    transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize([240,320]),
                                  transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),
                                  transforms.Normalize((0.5,),(0.5,))
                                 ])
    print(path)
    client=start_tcp_client(sys.argv[1],int(sys.argv[2]))
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path,f)):
            f_text, f_ext= os.path.splitext(f)
            f_ext= f_ext[1:].upper()
            
            if f_ext in extensions:# and f[0]=='r':
                print(f)
                #img = Image.open(os.path.join(path,f)).resize((224,224))
                
                #client=start_tcp_client('127.0.0.1',4445)
                img=plt.imread(os.path.join(path,f))
                #print(type(img),img.shape)
                #img=img.reshape((240*320))
                
                img=transform(img).numpy()
                img=img.reshape((240,320))
                norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                norm_image=norm_image.astype(np.uint8)
                #print(type(norm_image[0][0]))
                #lt.imshow(norm_image, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                #lt.show()
                
                msg=np.array([random.randint(0,2)],dtype='uint8')
                #print(msg)
                msg=np.append(msg,norm_image)
                #print(len(msg),type(msg[0]))
                et=1234
                e1=np.uint8(et)
                e2=np.uint8(et>>8)
                e3=np.uint8(et>>16)
                e4=np.uint8(et>>24)
                dt=5678
                d1=np.uint8(dt)
                d2=np.uint8(dt>>8)
                d3=np.uint8(dt>>16)
                d4=np.uint8(dt>>24)
                msg=np.concatenate((msg,e4,e3,e2,e1,d4,d3,d2,d1), axis=None)
                
                #print(msg)
#                 b=struct.pack('I',1234)
#                 print(b,type(b),len(b))
#                 msg=np.append(msg,b)      
#                 b=struct.pack('I',5678)
#                 print(b,type(b),len(b))
#                 msg=np.append(msg,b) 
                #print(len(msg),type(msg[0]))
                client.send(msg)
                '''
                data = pickle.dumps(img, 0)
                #print(data)
                size = len(data)
                print("{}".format(size))
                client.sendall(struct.pack(">L", size) + data)
                '''
                rcv_msg=client.recv(1024)
                #print(rcv_msg,len(rcv_msg))
                data = np.fromstring(rcv_msg, dtype='uint8')
                print(data)
    client.close()
                #time.sleep(1)
