from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import entropyCalculation

import pandas as pd

import csv

import numpy as np
obj = entropyCalculation.ddosDetection()
csv_file_path = 'data/datasetWithLabel.csv'  
import update
from datetime import datetime, timedelta
def cal_window(time):
    # 示例时间字符串
    # time_str = "5/7/2017 3:12"

    # 将时间字符串解析为datetime对象
    datetime_obj = datetime.strptime(time, "%m/%d/%Y %H:%M")

    # 窗口间隔（1分钟）
    window_interval = timedelta(minutes=1)

    # 初始窗口起始时间
    window_start_time = datetime_obj.replace(second=0, microsecond=0)

    # 存储窗口计数的字典
    window_counts = {}

    # 模拟数据，假设有10个时间字符串
    for _ in range(10):
        # 更新当前时间
        current_time = window_start_time + window_interval

        # 计算当前窗口的键
        window_key = current_time.strftime("%m/%d/%Y %H:%M")

        # 更新窗口计数
        window_counts[window_key] = window_counts.get(window_key, 0) + 1

        # 移动到下一个窗口
        window_start_time = current_time

    # 打印窗口计数结果
    for window_key, count in window_counts.items():
        print(f"Window: {window_key}, Count: {count}")





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
        epoch = row[epoch_index]
        ip_src = row[ip_src_index]
        label=row[label_index]
        time=row[time_index]

        # 在这里进行你的处理，比如打印或其他操作
        # print(f'Epoch: {epoch}, IP_SRC: {ip_src}, Label:{label}')
        
        obj.get_ddos_list(time,label)
        #print "Entropy value = ",str(obj.sumEntropy)
        # obj.calculateEntropy(ip_src,time,0.37)
        # if (obj.ddosDetected == 1) :
        #     print("Controller detected DDOS ATTACK, Entropy value :",str(obj.sumEntropy))
        #     obj.ddosDetected = 0
        #     print("Future work to implement prevention methods")
        #     break
obj.idx=0
obj.flag=0
obj.start_time=""
threshold=0.37   
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)

    # 获取列索引
    header = next(csv_reader)
    epoch_index = header.index('epoch')
    time_index = header.index('time')
    ip_src_index = header.index('ip_src')
    label_index = header.index('label')
    for row in csv_reader:
        epoch = row[epoch_index]
        ip_src = row[ip_src_index]
        label=row[label_index]
        time=row[time_index]

        # 在这里进行你的处理，比如打印或其他操作
        # print(f'Epoch: {epoch}, IP_SRC: {ip_src}, Label:{label}')
        
        # obj.get_ddos_list(time,label)
        #print "Entropy value = ",str(obj.sumEntropy)
        obj.calculateEntropy(ip_src,time,threshold)
        # if (obj.ddosDetected == 1) :
        #     print("Controller detected DDOS ATTACK, Entropy value :",str(obj.sumEntropy))
        #     obj.ddosDetected = 0
        #     print("Future work to implement prevention methods")
        #     break
        if len(obj.detection_win) == 15:
            flag=1
            for i,value in enumerate(obj.detection_win):
                if value>0:
                    flag=0
                    break
            if flag:
                threshold=update.chebyshev_inequality(obj.EV,4)
            print(obj.detection_win,threshold)
            obj.detection_win = np.array([])
        if len(obj.EV) >= 15:
            # 当数组元素个数达到 10 时删除第一个元素
            obj.EV = np.delete(obj.EV, 0)
    print(obj.ddos_dist)
    # print(obj.norm_dist)
    # print(obj.idx)
    # print(sum(obj.ddos_dist))
    print(obj.detection_dist)
# k=4
# Confusion Matrix:
# [[454  36]
#  [  0  18]]
# 0.9291338582677166 1.0 0.07346938775510205 0.3333333333333333
# k=6
# Confusion Matrix:
# [[461  29]
#  [  8  10]]
# 0.9271653543307087 0.5555555555555556 0.05918367346938776 0.2564102564102564
# k=8
# Confusion Matrix:
# [[466  24]
#  [ 16   2]]
# 0.9212598425196851 0.1111111111111111 0.04897959183673469 0.07692307692307693
# k=1
# Confusion Matrix:
# [[310 180]
#  [  0  18]]
# 0.6456692913385826 1.0 0.3673469387755102 0.09090909090909091
# 计算混淆矩阵
conf_matrix = confusion_matrix(obj.ddos_dist[:-1], obj.detection_dist)
tn, fp, fn, tp = np.array(conf_matrix).ravel()

# 输出混淆矩阵
print("Confusion Matrix:")
print(conf_matrix)
# 计算正确率
accuracy = accuracy_score(obj.ddos_dist[:-1], obj.detection_dist)
dr=tp/(tp+fn)
fpr=fp/(fp+tn)
pr=tp/(tp+fp)
print(accuracy,dr,fpr,pr)