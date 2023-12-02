from cProfile import label
import math
from re import S
from matplotlib.pyplot import flag

import pandas as pd

# 手动给一个阈值，根据时间窗口计算ip的信息熵，归一化之后与阈值做比较，超出阈值则视为存在攻击
# （这里不纠结信息熵的计算方式，统一用香农公式）
# 如果时间窗口内没有检测出DDoS攻击，则更新阈值
# 策略如下：
# 1.霍夫丁不等式
# 2.强化学习



class ddosDetection:
   # 从时间窗口1开始检测，一次检测6个窗口（即30min），如果超过半数的窗口的熵值低于阈值则判定存在ddos攻击
   # 这个策略需要根据实际数据来定，在ddos窗口计算分布情况，再根据这个改变策略
   time_win=1
   pktCnt = 0
   #ddosDetected : 1 indicates ddosDetected is true , 0 : false
   ddosDetected = 0
   # if 10 times consecutively , entropy value is less than 1, then indicate to controller than DDOS Attack is detected
   counter = 0
   ipList_Dict = {}
   sumEntropy = 0
   ddos_dist=[]
   idx=0
   ddos_dist.append(0)
   res_dist={}
   norm_dist={}
   detection_dist=[]
   start_time=""
   flag=0
   def calculateEntropy(self,ip,time,label):
      #calculate entropy when pkt cont reaches 100
      self.pktCnt +=1
      if ip in self.ipList_Dict:
         self.ipList_Dict[ip] += 1
      else:
         self.ipList_Dict[ip] = 0
      if pd.notna(label) and label=='DoS Hulk':
         self.ddos_dist[self.idx]=1
      if self.flag==0:
         self.start_time=time
         self.flag=1
      # if self.pktCnt == 500:
      if time!=self.start_time:
         
         
         #print self.ipList_Dict.items()
         print( self.pktCnt)
         self.sumEntropy = 0
         self.ddosDetected = 0
         print( "Window size of 50 pkts reached, calculate entropy")
         # //计算香农熵
         for ip,value in self.ipList_Dict.items():
            prob = abs(value/float(self.pktCnt))
            #print prob
            if (prob > 0.0) : 
               ent = -prob * math.log(prob,2)
               #print ent
               self.sumEntropy =  self.sumEntropy + ent
         self.res_dist[self.idx]=self.sumEntropy
         # print( "Entropy Value = ",self.sumEntropy ) 
         # 归一化香农熵到 [0, 1] 范围
         max_possible_entropy = -math.log(1 / float(len(self.ipList_Dict)), 2)
         normalized_entropy = self.sumEntropy / max_possible_entropy
         self.norm_dist[self.idx]=normalized_entropy
         print("Normalized Entropy Value =", normalized_entropy)
         # print(self.ddos_dist)
         # print(self.res_dist)
         # 目测阈值设置为1.6可以准确检测出dos hulk攻击
         # 1.6:0.8867924528301887
         
         # 1.7:0.9311023622047244
#          Confusion Matrix:
# [[468  22]
#  [ 13   5]]
         # if (self.sumEntropy < 1.7 and self.sumEntropy != 0) :
         # 0.5:0.8681102362204725
         # 0.4:0.9291338582677166
         # Confusion Matrix:
         # [[454  36]
         #  [  0  18]]
         # 0.3:0.9232283464566929
         # Confusion Matrix:
         # [[460  30]
         #  [  9   9]]
         # 0.36:0.9350393700787402
         # Confusion Matrix:
         # [[458  32]
         # [  1  17]]
         if 0 < normalized_entropy < 0.37:
            # self.counter += 1
            self.detection_dist.append(1)
         else :
            self.detection_dist.append(0)
         self.idx=self.idx+1
         self.ddos_dist.append(0)
         # if self.counter == 10:
         #    self.ddosDetected = 1
         #    print( "Counter = ",self.counter)
         #    print( "DDOS ATTACK DETECTED")
         #    self.counter = 0 
         self.cleanUpValues()
      # else:
         self.flag=0
      #    print("this epoch is done,please turn to next epoch")       
      
   def cleanUpValues(self):
      self.pktCnt = 0
      self.dest_ipList = []
      self.ipList_Dict = {}
      self.sumEntropy = 0
      # self.ddos_dist={}
      # self.idx=0
      # self.ddos_dist[self.idx]=0
      # self.res_dist={}

def __init__(self):
    pass
