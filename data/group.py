import pandas as pd
from datetime import datetime
from collections import Counter
import math

# 定义计算信息熵的函数
def calculate_entropy(ip_src_counts):
    total_count = sum(ip_src_counts.values())
    if total_count == 0:
        return 0
    entropy = -sum((count / total_count) * math.log2(count / total_count) for count in ip_src_counts.values() if count > 0)
    if math.log2(len(ip_src_counts)) == 0:
        return 1
    return entropy / math.log2(len(ip_src_counts))

def cal_entropy():
    # 读取CSV文件
    # df = pd.read_csv('csv\epoch_26.csv')
    df = pd.read_csv('data\datasetWithLabel.csv')
    # 将时间列转换为 datetime 类型
    # df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y %H:%M')


    # 根据每分钟的时间分组数据
    grouped_data = df.groupby('time')
    # print(grouped_data)
    # 初始化结果字典
    result_dict = {'time': [], 'normalized_entropy': []}

    # 遍历每分钟的数据
    for name, group in grouped_data:
        # 计算每分钟的ip_src信息熵
        ip_src_counts = Counter(group['ip_src'])
        print(ip_src_counts)
        entropy = calculate_entropy(ip_src_counts)
        print(entropy)
        # 归一化信息熵
        # normalized_entropy = entropy / math.log2(len(ip_src_counts))
        
        # 将结果添加到字典
        result_dict['time'].append(name)
        result_dict['normalized_entropy'].append(entropy)

    # 将结果转换为DataFrame
    result_df = pd.DataFrame(result_dict)
    # 将结果写入新的CSV文件
    result_df.to_csv('entropy.csv', index=False)

def cal_ddos():
    # df = pd.read_csv('csv\epoch_26.csv')
    # 读取CSV文件
    df = pd.read_csv('data\datasetWithLabel.csv')

    # 转换时间列为datetime类型
    # df['time'] = pd.to_datetime(df['time'])
    # 将 'time' 列转换为 datetime 类型
    # df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M')


    def check_non_benign(group):
        # Check if any value in the 'label' column is not NaN and not 'BENIGN'
        if any(group['label'].notna() & (group['label'] != 'BENIGN')):
            # Print the non-'BENIGN' value
            print(group.loc[group['label'].notna() & (group['label'] != 'BENIGN'), 'label'].iloc[0])
            return pd.Series({'DDOS': 1})
        else:
            return pd.Series({'DDOS': 0})
    # 根据'time'列分组并应用函数
    result_df = df.groupby('time').apply(check_non_benign).reset_index()

    # 将结果写入新的CSV文件
    result_df.to_csv('ddos.csv', index=False)

# cal_entropy()
cal_ddos()