#!/usr/bin/python

import sys
import socket
import threading
import signal
import datetime
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import struct

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import namedtuple

from socket_me import start_tcp_server, start_tcp_client

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, QMutex
from PyQt5.QtGui import QImage

import cv2
from PIL.ImageQt import ImageQt
import PIL
''' 
    packet_array=[plate_id, gray level image, eating time, drinking time]

                   1 byte ,   320x240 bytes ,    4 bytes ,    4 bytes

                   uint8_t,     uint8_t[]   ,    uint32_t,    uint32_t
'''
IMG_RESOLUTION              = 320 * 240
ADDR_OFFSET_PLATE_ID        = 0
ADDR_OFFSET_IMAGE           = 1
ADDR_OFFSET_EATING_TIME     = 1 + IMG_RESOLUTION
ADDR_OFFSET_DRINKING_TIME   = 1 + IMG_RESOLUTION + 4
LENGTH_PACKET               = ADDR_OFFSET_DRINKING_TIME + 4 


FIG_SIZE=224
ui_mutex=QMutex()
dog_mutex=QMutex()

'''
def start_tcp_server(ip, port, listen_num):
    #create socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as error_msg:
        print ("fail to listen on port %s"%error_msg)
        sys.exit(1)
    #ip = socket.gethostbyname(socket.gethostname())
    server_address = (ip, port)
    #bind port
    print('starting listen on ip %s, port %s'%server_address)
    sock.bind(server_address)
    #starting listening, allow only one connection
    sock.listen(listen_num)
    return sock

def start_tcp_client(server_ip, server_port):
    #create socket
    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        tcp_client.connect((server_ip, server_port))
    except socket.error as error_msg:
        print('fail to setup socket connection %s'%error_msg)
    return tcp_client
'''

class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
                
        block, n_blocks, channels = config
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h
    
class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x
    
class Dog_user:
    def __init__(self, dogid):
        self.dog_id=dogid
        self.dog_place="placeA"
        self.dog_eat_time=0
        self.dog_drink_time=0
        self.dog_avg_eat_time=0
        self.dog_avg_drink_time=0
    def show_info(self):
        print("dog%d at %s, eat %d / %d secs, drink %d / %d secs"%(self.dog_id,self.dog_place,self.dog_eat_time,self.dog_avg_eat_time,self.dog_drink_time,self.dog_avg_drink_time))
        

class Dog_Classification:
    def __init__(self):
        ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
        resnet18_config = ResNetConfig(block = BasicBlock,n_blocks = [2,2,2,2],channels = [64, 128, 256, 512])
        self.OUTPUT_DIM = 4
        self.model = ResNet(resnet18_config, self.OUTPUT_DIM)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        #weights = [1/55/5, 1/86/5, 1/126/5, 1/459/5] #[ 1 / number of instances for each class]
        #self.class_weights = torch.FloatTensor(weights)
        #self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #model = model.to(device)
        #criterion = criterion.to(device)
        print(self.model)
        self.model.load_state_dict(torch.load('dog_model18_face.pt',map_location=torch.device('cpu')))
        self.model.eval()
        self.transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize([FIG_SIZE,FIG_SIZE]),transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5,),(0.5,0.5,0.5,))])
        #self.transform=transforms.Compose([transforms.Resize([FIG_SIZE,FIG_SIZE]),transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5,),(0.5,0.5,0.5,))])
        
    
    def crop_left1(self,img):
        #return img[:,0:149,:]
        return img[:,0:122,:]
    def crop_right1(self,img):
        #return img[:,75:224,:]
        return img[:,122:224,:]
    def crop_left2(self,img):
        z=np.zeros((224, 75, 3),dtype=np.uint8)
        rimg=np.append(img[:,0:149,:],z,axis=1)
        return rimg
    def crop_right2(self,img):
        z=np.zeros((224, 75, 3),dtype=np.uint8)
        rimg=np.append(z,img[:,75:224,:],axis=1)
        return rimg
        
    def test(self,img):
        #print(type(img),type(img[0]),type(img[0][0][0]))
        gray=self.transform(img)
        gray=gray.unsqueeze(0)
        l=self.crop_left1(img)
        gray_l=self.transform(l)
        gray_l=gray_l.unsqueeze(0)
        r=self.crop_right1(img)
        gray_r=self.transform(r)
        gray_r=gray_r.unsqueeze(0)
        
        gray=torch.cat((gray,gray_l,gray_r),0)
        
        logps, _ = self.model.forward(gray)
        ps = torch.exp(logps)
        _, top_class = ps.topk(1, dim=1)
        print(ps,top_class,top_class.shape)
        ifeat=top_class[1,0].numpy()==top_class[0,0].numpy() and ps[1,top_class[0,0].numpy()]>=ps[2,top_class[0,0].numpy()]
        ifdrink=top_class[2,0].numpy()==top_class[0,0].numpy() and ps[1,top_class[0,0].numpy()]<ps[2,top_class[0,0].numpy()]
        return top_class[0,0].numpy(), ifeat, ifdrink

    
class Fig_Server(QThread):
    sinOut = pyqtSignal(Dog_user)
    sinfigOut = pyqtSignal(np.ndarray,np.uint8)
    def __init__(self, parent=None):
        super(Fig_Server, self).__init__(parent)
        self.nn=Dog_Classification()
        self.sock = start_tcp_server(sys.argv[1], int(sys.argv[2]), 30)
        # self.fig_length=224*224*3
        self.fig_length = 320*240
        self.dog=[Dog_user(i) for i in range(3)]
        self.day=0
        self.date=datetime.datetime.now().day
    
    def change_day_timer(self):
        while True:
            if self.date!=datetime.datetime.now().day:
                self.date=datetime.datetime.now().day
                self.day+=1
                dog_mutex.lock()
                for i in range(3):
                    self.dog[i].dog_avg_eat_time=(self.dog[i].dog_avg_eat_time*(self.day-1)+self.dog[i].dog_eat_time)/self.day
                    self.dog[i].dog_avg_drink_time=(self.dog[i].dog_avg_drink_time*(self.day-1)+self.dog[i].dog_drink_time)/self.day
                    self.dog[i].dog_eat_time=0
                    self.dog[i].dog_drink_time=0
                    self.sinOut.emit(self.dog[i])
                dog_mutex.unlock()
            time.sleep(60)
                
        
        
    def run(self):
        print("i'm running")
        timer_handler=threading.Thread(target=self.change_day_timer, args=())
        timer_handler.start()
        threads = []
        while True:
            client,addr = self.sock.accept()

            print ("[*] Acepted connection from: %s:%d" % (addr[0],addr[1]))
            #fig_server.client_fig_handler(client)
            client_handler = threading.Thread(target=self.client_fig_handler, args=(client,))
            client_handler.start()
            threads.append(client_handler);

        self.sock.close()
        for thread in threads:
            thread.join()
        
        
    def recvall(self, conn, count):
        buf = b''
        while count:
            try:
                newbuf = conn.recv(count)

            except socket.error as error_msg: 
                print('samliu socket.error - ', error_msg)
                return None
            # if newbuf == b'':
            #     raise RuntimeError("socket connection broken-sammmmmm")
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def client_fig_handler(self, client_socket):
        '''
        data = b""
        payload_size = struct.calcsize(">L")
        print("payload_size: {}".format(payload_size))

        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += client_socket.recv(4096)

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += client_socket.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        img=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        '''

        while True:
            string_data = self.recvall(client_socket, LENGTH_PACKET)
            # string_data = recvall(conn, int(length))
            
            if not string_data:
                print('ERROR: Broken pipe while message receiving')
                break
            elif len(string_data) != LENGTH_PACKET:
                print('ERROR: Length of received data is {}, the expected length is {}', len(string_data), LENGTH_PACKET)
                break
            # print(len(string_data))
            # string_data = np.fromstring(string_data, dtype='bytes')
            packet_data = np.fromstring(string_data, dtype='uint8')
            #print(len(packet_data), packet_data.shape)

            # Plate ID data
            place_id = packet_data[ADDR_OFFSET_PLATE_ID]
            #print(f"placeid={place_id}")
            
            # Image data
            img_gray = packet_data[ADDR_OFFSET_IMAGE:ADDR_OFFSET_IMAGE + IMG_RESOLUTION]
            img_gray = img_gray.reshape((240, 320, 1))
            self.sinfigOut.emit(img_gray,place_id)

            # Eating time data
            tmp = packet_data[ADDR_OFFSET_EATING_TIME: ADDR_OFFSET_EATING_TIME + 4]
            eat_time = tmp[0]<<24 | tmp[1]<<16 | tmp[2]<<8 | tmp[3]<<0

            # Drinking time data
            tmp = packet_data[ADDR_OFFSET_DRINKING_TIME: ADDR_OFFSET_DRINKING_TIME + 4]
            drink_time = tmp[0]<<24 | tmp[1]<<16 | tmp[2]<<8 | tmp[3]<<0
            

            
            
            #print(f"placeid={place_id} eat_time={eat_time} drink_time={drink_time} ")

            # Image preprocessing
            img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
            img = np.zeros((240, 320, 3), dtype='uint8')
            img[:, :, 0] = img_gray
            img[:, :, 1] = img_gray
            img[:, :, 2] = img_gray
            #img=img_gray

            result,eat,drink=self.nn.test(img)
            
            if result!=3:
                dog_mutex.lock()
                if place_id==0:
                    self.dog[result].dog_place='place_A'
                elif place_id==1:
                    self.dog[result].dog_place='place_B'
                elif place_id==2:
                    self.dog[result].dog_place='place_C'

                self.dog[result].dog_eat_time+=eat_time+1
                self.dog[result].dog_drink_time+=drink_time+2
                self.dog[result].show_info()
                dog_mutex.unlock()
                self.sinOut.emit(self.dog[result])
            
            
            send_msg=np.array([result,eat,drink],dtype=np.uint8)
            #client_socket.send(("%d %d %d"%(result,eat,drink)).encode())
            try:
                client_socket.send(send_msg)
            except ConnectionResetError as error_msg:
                print('ERROR: Broken pipe while message sending')
                break


        client_socket.close()

#fig_server=Fig_Server()        

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

    
    
class Show_video(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self)
        self.setupUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(364, 303)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.dog_fig = QtWidgets.QLabel(self.centralWidget)
        self.dog_fig.setGeometry(QtCore.QRect(30, 20, 321, 241))
        self.dog_fig.setScaledContents(True)
        self.dog_fig.setObjectName("dog_fig")
        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Show_video"))
        self.dog_fig.setText(_translate("MainWindow", "dog video"))

        
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        
        self.thread = Fig_Server()
        self.thread.sinOut.connect(self.update_dog_info)
        self.thread.sinfigOut.connect(self.update_fig)
        self.thread.start()


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1127, 605)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.dog_name_1 = QtWidgets.QLabel(self.centralWidget)
        self.dog_name_1.setGeometry(QtCore.QRect(200, 100, 67, 17))
        self.dog_name_1.setObjectName("dog_name_1")
        self.dog_place_1 = QtWidgets.QTextBrowser(self.centralWidget)
        self.dog_place_1.setGeometry(QtCore.QRect(110, 330, 256, 31))
        self.dog_place_1.setObjectName("dog_place_1")
        self.dog_today_eat_time_1 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_eat_time_1.setGeometry(QtCore.QRect(110, 410, 81, 21))
        self.dog_today_eat_time_1.setObjectName("dog_today_eat_time_1")
        self.label_1_3 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_3.setGeometry(QtCore.QRect(200, 410, 51, 21))
        self.label_1_3.setObjectName("label_1_3")
        self.dog_avg_eat_time_1 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_eat_time_1.setGeometry(QtCore.QRect(253, 410, 71, 23))
        self.dog_avg_eat_time_1.setObjectName("dog_avg_eat_time_1")
        self.label_1_4 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_4.setGeometry(QtCore.QRect(330, 410, 41, 21))
        self.label_1_4.setObjectName("label_1_4")
        self.dog_avg_drink_time_1 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_drink_time_1.setGeometry(QtCore.QRect(253, 440, 71, 23))
        self.dog_avg_drink_time_1.setObjectName("dog_avg_drink_time_1")
        self.label_1_6 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_6.setGeometry(QtCore.QRect(330, 440, 41, 21))
        self.label_1_6.setObjectName("label_1_6")
        self.label_1_5 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_5.setGeometry(QtCore.QRect(200, 440, 51, 21))
        self.label_1_5.setObjectName("label_1_5")
        self.dog_today_drink_time_1 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_drink_time_1.setGeometry(QtCore.QRect(110, 440, 81, 21))
        self.dog_today_drink_time_1.setObjectName("dog_today_drink_time_1")
        self.label_1_1 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_1.setGeometry(QtCore.QRect(110, 380, 67, 17))
        self.label_1_1.setObjectName("label_1_1")
        self.label_1_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_2.setGeometry(QtCore.QRect(250, 380, 67, 17))
        self.label_1_2.setObjectName("label_1_2")
        self.label_eat = QtWidgets.QLabel(self.centralWidget)
        self.label_eat.setGeometry(QtCore.QRect(40, 410, 67, 17))
        self.label_eat.setObjectName("label_eat")
        self.label_drink = QtWidgets.QLabel(self.centralWidget)
        self.label_drink.setGeometry(QtCore.QRect(40, 440, 67, 17))
        self.label_drink.setObjectName("label_drink")
        self.label_1_7 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_7.setGeometry(QtCore.QRect(527, 410, 51, 21))
        self.label_1_7.setObjectName("label_1_7")
        self.dog_place_2 = QtWidgets.QTextBrowser(self.centralWidget)
        self.dog_place_2.setGeometry(QtCore.QRect(437, 330, 256, 31))
        self.dog_place_2.setObjectName("dog_place_2")
        self.label_1_8 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_8.setGeometry(QtCore.QRect(657, 440, 41, 21))
        self.label_1_8.setObjectName("label_1_8")
        self.label_1_9 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_9.setGeometry(QtCore.QRect(577, 380, 67, 17))
        self.label_1_9.setObjectName("label_1_9")
        self.dog_name_2 = QtWidgets.QLabel(self.centralWidget)
        self.dog_name_2.setGeometry(QtCore.QRect(527, 100, 67, 17))
        self.dog_name_2.setObjectName("dog_name_2")
        self.dog_today_eat_time_2 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_eat_time_2.setGeometry(QtCore.QRect(437, 410, 81, 21))
        self.dog_today_eat_time_2.setObjectName("dog_today_eat_time_2")
        self.dog_avg_drink_time_2 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_drink_time_2.setGeometry(QtCore.QRect(580, 440, 71, 23))
        self.dog_avg_drink_time_2.setObjectName("dog_avg_drink_time_2")
        self.label_1_10 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_10.setGeometry(QtCore.QRect(437, 380, 67, 17))
        self.label_1_10.setObjectName("label_1_10")
        self.dog_avg_eat_time_2 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_eat_time_2.setGeometry(QtCore.QRect(580, 410, 71, 23))
        self.dog_avg_eat_time_2.setObjectName("dog_avg_eat_time_2")
        self.label_1_11 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_11.setGeometry(QtCore.QRect(527, 440, 51, 21))
        self.label_1_11.setObjectName("label_1_11")
        self.dog_today_drink_time_2 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_drink_time_2.setGeometry(QtCore.QRect(437, 440, 81, 21))
        self.dog_today_drink_time_2.setObjectName("dog_today_drink_time_2")
        self.label_1_12 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_12.setGeometry(QtCore.QRect(657, 410, 41, 21))
        self.label_1_12.setObjectName("label_1_12")
        self.label_1_13 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_13.setGeometry(QtCore.QRect(860, 410, 51, 21))
        self.label_1_13.setObjectName("label_1_13")
        self.dog_place_3 = QtWidgets.QTextBrowser(self.centralWidget)
        self.dog_place_3.setGeometry(QtCore.QRect(770, 330, 256, 31))
        self.dog_place_3.setObjectName("dog_place_3")
        self.label_1_14 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_14.setGeometry(QtCore.QRect(990, 440, 41, 21))
        self.label_1_14.setObjectName("label_1_14")
        self.label_1_15 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_15.setGeometry(QtCore.QRect(910, 380, 67, 17))
        self.label_1_15.setObjectName("label_1_15")
        self.dog_name_3 = QtWidgets.QLabel(self.centralWidget)
        self.dog_name_3.setGeometry(QtCore.QRect(860, 100, 67, 17))
        self.dog_name_3.setObjectName("dog_name_3")
        self.dog_today_eat_time_3 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_eat_time_3.setGeometry(QtCore.QRect(770, 410, 81, 21))
        self.dog_today_eat_time_3.setObjectName("dog_today_eat_time_3")
        self.dog_avg_drink_time_3 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_drink_time_3.setGeometry(QtCore.QRect(913, 440, 71, 23))
        self.dog_avg_drink_time_3.setObjectName("dog_avg_drink_time_3")
        self.label_1_16 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_16.setGeometry(QtCore.QRect(770, 380, 67, 17))
        self.label_1_16.setObjectName("label_1_16")
        self.dog_avg_eat_time_3 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_eat_time_3.setGeometry(QtCore.QRect(913, 410, 71, 23))
        self.dog_avg_eat_time_3.setObjectName("dog_avg_eat_time_3")
        self.label_1_17 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_17.setGeometry(QtCore.QRect(860, 440, 51, 21))
        self.label_1_17.setObjectName("label_1_17")
        self.dog_today_drink_time_3 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_drink_time_3.setGeometry(QtCore.QRect(770, 440, 81, 21))
        self.dog_today_drink_time_3.setObjectName("dog_today_drink_time_3")
        self.label_1_18 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_18.setGeometry(QtCore.QRect(990, 410, 41, 21))
        self.label_1_18.setObjectName("label_1_18")
        self.dog_fig_1 = QtWidgets.QLabel(self.centralWidget)
        self.dog_fig_1.setGeometry(QtCore.QRect(110, 130, 251, 171))
        self.dog_fig_1.setText("")
        self.dog_fig_1.setPixmap(QtGui.QPixmap("dog1.jpg"))
        self.dog_fig_1.setScaledContents(True)
        self.dog_fig_1.setWordWrap(False)
        self.dog_fig_1.setObjectName("dog_fig_1")
        self.dog_fig_2 = QtWidgets.QLabel(self.centralWidget)
        self.dog_fig_2.setGeometry(QtCore.QRect(440, 130, 251, 171))
        self.dog_fig_2.setText("")
        self.dog_fig_2.setPixmap(QtGui.QPixmap("dog2.jpg"))
        self.dog_fig_2.setScaledContents(True)
        self.dog_fig_2.setObjectName("dog_fig_2")
        self.dog_fig_3 = QtWidgets.QLabel(self.centralWidget)
        self.dog_fig_3.setGeometry(QtCore.QRect(770, 130, 251, 171))
        self.dog_fig_3.setText("")
        self.dog_fig_3.setPixmap(QtGui.QPixmap("dog3.jpg"))
        self.dog_fig_3.setScaledContents(True)
        self.dog_fig_3.setObjectName("dog_fig_3")
        
        #self.pushButton1 = QtWidgets.QPushButton()
        #self.pushButton1.setGeometry(QtCore.QRect(110, 470, 81, 21))
        #self.pushButton1.setObjectName("dog1_video")
        
        self.pushButton1 = QtWidgets.QPushButton(self.centralWidget)
        #self.pushButton1.setGeometry(QtCore.QRect(110, 470, 251, 21))
        self.pushButton1.setGeometry(QtCore.QRect(20, 20, 50, 20))
        self.pushButton1.setObjectName("pushButton1")
        self.pushButton2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton2.setGeometry(QtCore.QRect(20, 40, 50, 20))
        #self.pushButton2.setGeometry(QtCore.QRect(440, 470, 251, 21))
        self.pushButton2.setObjectName("pushButton2")
        self.pushButton3 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton3.setGeometry(QtCore.QRect(20, 60, 50, 20))
        #self.pushButton3.setGeometry(QtCore.QRect(780, 470, 251, 21))
        self.pushButton3.setObjectName("pushButton3")
        
        self.pushButton1.clicked.connect(self.pushButton1_clicked)
        self.pushButton2.clicked.connect(self.pushButton2_clicked)
        self.pushButton3.clicked.connect(self.pushButton3_clicked)
        self.video1 = Show_video()
        self.video2 = Show_video()
        self.video3 = Show_video()
        
        MainWindow.setCentralWidget(self.centralWidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DOG"))
        self.dog_name_1.setText(_translate("MainWindow", "Dog1"))
        self.label_1_3.setText(_translate("MainWindow", "secs  /"))
        self.label_1_4.setText(_translate("MainWindow", "secs"))
        self.label_1_6.setText(_translate("MainWindow", "secs"))
        self.label_1_5.setText(_translate("MainWindow", "secs /"))
        self.label_1_1.setText(_translate("MainWindow", "Today"))
        self.label_1_2.setText(_translate("MainWindow", "Average"))
        self.label_eat.setText(_translate("MainWindow", "Eat"))
        self.label_drink.setText(_translate("MainWindow", "Drink"))
        self.label_1_7.setText(_translate("MainWindow", "secs  /"))
        self.label_1_8.setText(_translate("MainWindow", "secs"))
        self.label_1_9.setText(_translate("MainWindow", "Average"))
        self.dog_name_2.setText(_translate("MainWindow", "Dog2"))
        self.label_1_10.setText(_translate("MainWindow", "Today"))
        self.label_1_11.setText(_translate("MainWindow", "secs /"))
        self.label_1_12.setText(_translate("MainWindow", "secs"))
        self.label_1_13.setText(_translate("MainWindow", "secs  /"))
        self.label_1_14.setText(_translate("MainWindow", "secs"))
        self.label_1_15.setText(_translate("MainWindow", "Average"))
        self.dog_name_3.setText(_translate("MainWindow", "Dog3"))
        self.label_1_16.setText(_translate("MainWindow", "Today"))
        self.label_1_17.setText(_translate("MainWindow", "secs /"))
        self.label_1_18.setText(_translate("MainWindow", "secs"))
        self.pushButton1.setText(_translate("MainWindow", "place1"))
        self.pushButton2.setText(_translate("MainWindow", "place2"))
        self.pushButton3.setText(_translate("MainWindow", "place3"))
        
        #self.menuDOG.setTitle(_translate("MainWindow", "DOG"))
        self.dog_today_eat_time_1.setDigitCount(8)
        self.dog_today_drink_time_1.setDigitCount(8)
        self.dog_avg_eat_time_1.setDigitCount(8)
        self.dog_avg_drink_time_1.setDigitCount(8)
        
    def pushButton1_clicked(self):
        print("click1")
        self.video1.show()
    def pushButton2_clicked(self):
        print("click2")
        self.video2.show()
    def pushButton3_clicked(self):
        print("click3")
        self.video3.show()

    def update_fig(self, img, place):
        
        qimage = QImage(img.data, 320, 240, QImage.Format_Grayscale8)
        #qim = ImageQt(img)
        pix = QtGui.QPixmap.fromImage(qimage)
        if place==0:
            self.video1.dog_fig.setPixmap(pix);
        elif place==1:
            self.video2.dog_fig.setPixmap(pix);
        elif place==2:
            self.video3.dog_fig.setPixmap(pix);
        
        
    def update_dog_info(self, msg):
        ui_mutex.lock()
        if msg.dog_id==0:
#             self.dog_today_eat_time_1.value=msg.dog_eat_time
#             self.dog_today_drink_time_1.value=msg.dog_drink_time
#             self.dog_avg_eat_time_1.value=msg.dog_avg_eat_time
#             self.dog_avg_drink_time_1.value=msg.dog_avg_drink_time
            self.dog_today_eat_time_1.display(msg.dog_eat_time)
            self.dog_today_drink_time_1.display(msg.dog_drink_time)
            self.dog_avg_eat_time_1.display(msg.dog_avg_eat_time)
            self.dog_avg_drink_time_1.display(msg.dog_avg_drink_time)
            self.dog_place_1.setText(msg.dog_place)
        elif msg.dog_id==1:
            self.dog_today_eat_time_2.display(msg.dog_eat_time)
            self.dog_today_drink_time_2.display(msg.dog_drink_time)
            self.dog_avg_eat_time_2.display(msg.dog_avg_eat_time)
            self.dog_avg_drink_time_2.display(msg.dog_avg_drink_time)
            self.dog_place_2.setText(msg.dog_place)
        elif msg.dog_id==2:
            self.dog_today_eat_time_3.display(msg.dog_eat_time)
            self.dog_today_drink_time_3.display(msg.dog_drink_time)
            self.dog_avg_eat_time_3.display(msg.dog_avg_eat_time)
            self.dog_avg_drink_time_3.display(msg.dog_avg_drink_time)
            self.dog_place_3.setText(msg.dog_place)

        ui_mutex.unlock()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR! Usage: python3 ui_server.py <server_ip> <server_port>")
        exit(-1)

    signal.signal(signal.SIGINT, signal_handler)
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    

    
    


