import csv
import unittest
from user_info import User
import matplotlib.pyplot as plt
from optimization import Optimization


def generate_od_pairs():
    net_xml_path = 'DCC.net.xml'
    source_edge = '361450282'
    target_edge = "-110407380#1"
    start_mode = 'walking'
    ant_num = [350]
    episodes = [500, 1000, 1500, 2000]
    iteration = 1
    db_path = 'test_new.db'
    user = User(60, True, 0, 20)

    optimization = Optimization(net_xml_path, user, db_path, source_edge,target_edge)  # 初始化Optimization类
    od_pairs = optimization.choose_od_pairs()  # 获取OD对

    with open('od_pairs.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for pair in od_pairs:
            writer.writerow(pair)  # 将每个OD对写入CSV文件

if __name__ == '__main__':
    generate_od_pairs()
