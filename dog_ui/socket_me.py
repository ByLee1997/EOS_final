#!/usr/bin/python

import sys
import socket

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