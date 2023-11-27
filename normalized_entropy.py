import numpy as np
from scipy.stats import entropy

def normalized_entropy(data):
    # 计算信息熵
    entropy_value = entropy(data, base=2)
    
    # 计算最大可能熵
    max_entropy = np.log2(len(data))
    
    # 计算归一化熵值
    normalized_entropy_value = entropy_value / max_entropy
    
    return normalized_entropy_value

# 示例数据
data = [0, 0, 1, 1, 1, 2, 2, 2, 2]

# 计算归一化熵值
ne_value = normalized_entropy(data)

print(f"归一化熵值：{ne_value}")
