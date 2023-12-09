
from torch import threshold
import numpy as np
import time
from cProfile import label
import math
from re import S
from matplotlib.pyplot import flag
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import csv
import entropyCalculation
from datetime import datetime, timedelta

class DDoS(object):
    def __init__(self):
        super(DDoS, self).__init__()
        self.action_space = ['increase', 'decrease', 'still']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.obj = entropyCalculation.ddosDetection()
        self.pre_accuracy=0
        self.base_threshold = 0.37
        self._build_env()


    def _build_env(self):
        csv_file_path = 'data\datasetWithLabel.csv'  
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            # 获取列索引
            header = next(csv_reader)
            epoch_index = header.index('epoch')
            time_index = header.index('time')
            ip_src_index = header.index('ip_src')
            label_index = header.index('label')
            
            # 遍历每一行
            for row in csv_reader:
                label=row[label_index]
                time=row[time_index]

                self.obj.get_ddos_list(time,label)

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        
        if action == 0:  
            base_threshold-=0.1
        elif action == 1:  
            base_threshold+=0.1
        csv_file_path = 'datasetWithLabel.csv'  
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            # 获取列索引
            header = next(csv_reader)
            epoch_index = header.index('epoch')
            time_index = header.index('time')
            ip_src_index = header.index('ip_src')
            label_index = header.index('label')
            
            # 遍历每一行
            for row in csv_reader:
                time=row[time_index]
                ip_src = row[ip_src_index]
                self.obj.calculateEntropy(ip_src,time,base_threshold)
        # 计算混淆矩阵
        # conf_matrix = confusion_matrix(self.obj.ddos_dist[:-1], self.obj.detection_dist)
        # 输出混淆矩阵
        print("Confusion Matrix:")
        # print(conf_matrix)
        # 计算正确率
        accuracy = accuracy_score(self.obj.ddos_dist[:-1], self.obj.detection_dist)
        # print(accuracy)

        # reward function
        if accuracy >self.pre_accuracy:
            reward = 1
            # done = True
        elif accuracy < self.pre_accuracy:
            reward = -1
            # done = True
        else:
            reward = 0
            # done = False
        if accuracy>0.95:
            done = True
        else:
            done = False
        
        self.pre_accuracy=accuracy
        return reward,done

    def render(self):
        # time.sleep(0.01)
        self.update()


