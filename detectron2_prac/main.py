"""
-*- coding: uft-8 -*-
@Description:
@Time: 2024/4/24 22:43
@Author: shaowei
"""
import socket
from urllib.parse import urlparse


def get_domain_and_ip(url):
    try:
        url = urlparse(url)
        domain = url.netloc or url.path
        print(f"Domain: {domain}")

        # 尝试使用HTTP和HTTPS获取地址信息
        addresses = socket.getaddrinfo(domain, 'http') + socket.getaddrinfo(domain, 'https')

        # 过滤出IPv4地址并打印
        for address in addresses:
            if address[0] == socket.AF_INET:
                print(f"IP: {address[4][0]}")
                break
    except Exception as e:
        print(f"Error: {str(e)}")


url = input("输入网站的url: ")
get_domain_and_ip(url)
