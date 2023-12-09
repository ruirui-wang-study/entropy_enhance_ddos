from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import entropyCalculation

import pandas as pd
import numpy as np
import update
from datetime import datetime, timedelta
class detection():
      
    def __init__(self) -> None:
        self.obj = entropyCalculation.ddosDetection()
        self.csv_file_path = 'data/datasetWithLabel.csv'
        self.data,self.epoch_index,self.time_index,self.ip_src_index,self.label_index=self.load_file(self.csv_file_path)
    
    
    def load_file(file):
        df = pd.read_csv(file)  # 使用Pandas读取CSV文件
        # 按照time列的值进行分组
        grouped_data = df.groupby('time')
        # 获取列索引
        epoch_index = df.columns.get_loc('epoch')
        time_index = df.columns.get_loc('time')
        ip_src_index = df.columns.get_loc('ip_src')
        label_index = df.columns.get_loc('label')
        return grouped_data,epoch_index,time_index,ip_src_index,label_index
        

    @classmethod
    def getDDoSList(self,group):        
        # 遍历每一行
        # for row in group.iterrows():
            # label=row[self.label_index]
            # time=row[self.time_index]

            # 在这里进行你的处理，比如打印或其他操作
            # print(f'Epoch: {epoch}, IP_SRC: {ip_src}, Label:{label}')
            
            self.obj.get_ddos_list(group)
            # self.obj.idx=0
            # self.obj.flag=0
            # self.obj.start_time=""
    
    @classmethod 
    def calculate_entropy(self,threshold):  
        EV = np.array([])
        detection_win=np.array([])      
        pktCnt=0
        ipList_Dict = {}
        detection_dist=[]

        for row in self.df.iterrows():
            ip = row[self.ip_src_index]
            time=row[self.time_index]
            pktCnt +=1
            if ip in ipList_Dict:
                ipList_Dict[ip] += 1
            else:
                ipList_Dict[ip] = 0

            if flag==0:
                start_time=time
                flag=1
            # print(time)
            if time!=self.start_time:

                self.obj.calculateEntropy(ip_src,time,threshold)
            
            if len(self.obj.detection_win) == 15:
                flag=1
                for i,value in enumerate(self.obj.detection_win):
                    if value>0:
                        flag=0
                        break
                if flag:
                    threshold=update.chebyshev_inequality(self.obj.EV,4)
                print(self.obj.detection_win,threshold)
                self.obj.detection_win = np.array([])
            if len(self.obj.EV) >= 15:
                # 当数组元素个数达到 10 时删除第一个元素
                self.obj.EV = np.delete(self.obj.EV, 0)
            self.obj.calculateEntropy(ip_src,time,0.37)
    # print(obj.ddos_dist)
    # print(obj.norm_dist)
    # print(obj.idx)
    # print(sum(obj.ddos_dist))
    # print(obj.detection_dist)

            
   
         
    def calculatemetrix(ddos_dist,detection_dist):
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(ddos_dist[:-1], detection_dist)
        tn, fp, fn, tp = np.array(conf_matrix).ravel()

        # 输出混淆矩阵
        print("Confusion Matrix:")
        print(conf_matrix)
        # 计算正确率
        accuracy = accuracy_score(ddos_dist[:-1], detection_dist)
        dr=tp/(tp+fn)
        fpr=fp/(fp+tn)
        pr=tp/(tp+fp)
        print(accuracy,dr,fpr,pr)
        
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