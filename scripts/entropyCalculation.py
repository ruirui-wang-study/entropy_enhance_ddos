from cProfile import label
import math
from re import S
# from matplotlib.pyplot import flag

import pandas as pd

import numpy as np

# 手动给一个阈值，根据时间窗口计算ip的信息熵，归一化之后与阈值做比较，超出阈值则视为存在攻击
# （这里不纠结信息熵的计算方式，统一用香农公式）
# 如果时间窗口内没有检测出DDoS攻击，则更新阈值
# 策略如下：
# 1.霍夫丁不等式
# 2.强化学习



class ddosDetection:
   # 从时间窗口1开始检测，一次检测6个窗口（即30min），如果超过半数的窗口的熵值低于阈值则判定存在ddos攻击
   # 这个策略需要根据实际数据来定，在ddos窗口计算分布情况，再根据这个改变策略


   def calculateEntropy(self,group,threshold):

      sumEntropy = 0
      res_dist=[]
      norm_dist=[]
      ip_counts = group['ip'].value_counts()
      # 计算总的IP个数
      total_ips = len(ip_counts)
      pktCnt=len(group)

      prob = abs(ip_counts/float(pktCnt))
      if (prob > 0.0) : 
         ent = -prob * math.log(prob,2)
         sumEntropy =  sumEntropy + ent
      res_dist.append(sumEntropy)
      # print( "Entropy Value = ",self.sumEntropy ) 
      # 归一化香农熵到 [0, 1] 范围
      max_possible_entropy = -math.log(1 / float(total_ips), 2)
      normalized_entropy = self.sumEntropy / max_possible_entropy
      norm_dist.append(normalized_entropy)

      detection_dist=[]
      if 0 < normalized_entropy < threshold:
         detection_dist.append(1)
      else :
         detection_dist.append(0)
  
      
   def get_ddos_list(self,group):
      label=group['label']
      ddos_list=[]
      if pd.notna(label) and label!='DoS Hulk':
         ddos_list.append(1)
      return ddos_list
     

def __init__(self):
    pass
