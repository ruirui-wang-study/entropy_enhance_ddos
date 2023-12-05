import math

def hoeffding_inequality_probability(n, epsilon):
    return 2 * math.exp(-2 * n * epsilon**2)

# 示例用法
sample_size = 1000  # 样本数
epsilon_value = 0.1  # 给定的正数

probability = hoeffding_inequality_probability(sample_size, epsilon_value)

print(f"The probability is approximately: {probability}")

# 这是霍夫丁不等式的一个应用，用于估计样本平均值与总体平均值之间的差异的概率。具体来说，这个不等式表达的是事件 \(P(\bar{S} - \mu \geq 2)\) 的概率上界，其中 \(\bar{S}\) 是样本均值，\(\mu\) 是总体均值。

# 首先，霍夫丁不等式的一般形式为：

# \[ P(\bar{S} - \mu \geq \epsilon) \leq e^{-2n\epsilon^2} \]

# 其中，\(\epsilon\) 是一个常数，表示样本均值与总体均值之间的差异，\(n\) 是样本数。

# 在给定的例子中，我们有 \(\epsilon = 2\)，\(n = 10\)（样本数为10），而不等式右侧的部分是 \(e^{-2n\epsilon^2}\)。

# 计算步骤如下：

# 1. 将 \(\epsilon\) 和 \(n\) 代入不等式：\[ e^{-2n\epsilon^2} = e^{-2 \times 10 \times 2^2} \]

# 2. 进行计算：\[ e^{-2 \times 10 \times 2^2} = e^{-8/10} \]

# 3. 得到结果：\[ e^{-8/10} \approx 0.45 \]

# 所以，不等式右侧的值约为0.45。这表示事件 \(P(\bar{S} - \mu \geq 2)\) 的概率上界为0.45。也就是说，样本均值与总体均值之间的差异大于等于2的概率上界为0.45。





# 霍夫丁不等式给出了一个随机变量与其期望之间偏离的概率上界，通常用于估计样本平均值与总体均值之间的差异。具体而言，霍夫丁不等式的形式为：

# \[ P\left(\left|\frac{1}{n}\sum_{i=1}^{n}X_i - \mu\right| \geq \epsilon\right) \leq 2e^{-2n\epsilon^2/(b-a)^2} \]

# 其中，\(X_1, X_2, \ldots, X_n\) 是独立同分布的随机变量，取值范围在 \([a, b]\) 区间内，\(\mu\) 是随机变量的均值，\(\epsilon > 0\) 是任意正数，\(n\) 是样本容量。

# 如果我们想要得到样本平均值与总体均值之间偏离的概率下界，可以通过对不等式取补集来得到：

# \[ P\left(\left|\frac{1}{n}\sum_{i=1}^{n}X_i - \mu\right| < \epsilon\right) \geq 1 - 2e^{-2n\epsilon^2/(b-a)^2} \]

# 这表示样本平均值与总体均值之间偏离小于 \(\epsilon\) 的概率至少为 \(1 - 2e^{-2n\epsilon^2/(b-a)^2}\)。

# 在实际计算中，你可以根据具体的问题选择合适的 \(\epsilon\) 和样本容量 \(n\)，然后计算右侧的值。这个值就是样本平均值与总体均值之间偏离小于 \(\epsilon\) 的概率的下界。





# 根据霍夫丁不等式计算信息熵的上界
# 设置窗口区间，计算信息熵平均值，利用霍夫丁不等式计算最新平均值和总体平均值之间的概率差，得出平均值的上下界，然后在这个窗口内判断ddos攻击
# 目前是每一分钟计算一次信息熵，可以设置窗口为10，每十分钟计算一次平均值与阈值做比较，判断这个窗口内是否有攻击
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



# 根据当前归一化熵值和最后N个检测区间的熵值数组，计算数组的平均值M和标准差gamma
# 阈值sigma=M-k*gamma
def chebyshev_inequality(data, k):
    """
    切比雪夫不等式：给出随机变量偏离其均值的上界。

    Parameters:
        - data: 包含随机变量数据的数组或列表
        - mean: 随机变量的均值
        - k: 切比雪夫不等式中的常数，表示偏离均值的程度

    Returns:
        - lower_bound: 偏离均值的下界
        - upper_bound: 偏离均值的上界
    """
    std_dev = np.std(data)
    mean = np.array(data)

    lower_bound = mean - k * std_dev
    upper_bound = mean + k * std_dev

    return lower_bound

# 示例数据
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 计算均值
mean_value = np.mean(data)

# 设置切比雪夫不等式中的常数 k
k_value = 2

# 计算切比雪夫不等式的上下界
lower, upper = chebyshev_inequality(data, mean_value, k_value)

print(f"切比雪夫不等式的下界：{lower}")
print(f"切比雪夫不等式的上界：{upper}")
