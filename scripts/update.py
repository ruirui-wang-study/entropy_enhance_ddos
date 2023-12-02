import math

def hoeffding_inequality_probability(n, epsilon):
    return 2 * math.exp(-2 * n * epsilon**2)

# 示例用法
sample_size = 1000  # 样本数
epsilon_value = 0.1  # 给定的正数

probability = hoeffding_inequality_probability(sample_size, epsilon_value)

print(f"The probability is approximately: {probability}")



# 根据霍夫丁不等式计算信息熵的上界
import numpy as np

def hoeffding_bound(sample_mean, true_mean, epsilon, n):
    bound = 2 * np.exp(-2 * n * epsilon**2)
    upper_bound = sample_mean + epsilon
    return upper_bound if bound > 1 else upper_bound + 1

# 生成二项分布的样本
n = 1000  # 样本大小
p = 0.3   # 成功的概率
sample = np.random.binomial(1, p, n)

# 计算样本平均值
sample_mean = np.mean(sample)

# 设置真实的平均值和误差
true_mean = p
epsilon = 0.1

# 计算霍夫丁上界
upper_bound = hoeffding_bound(sample_mean, true_mean, epsilon, n)

print(f"样本平均值: {sample_mean}")
print(f"霍夫丁上界: {upper_bound}")
