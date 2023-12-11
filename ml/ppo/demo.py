import gym
from gym import spaces
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

# 读取CSV文件
ddos_df = pd.read_csv('RL/ddos.csv')
entropy_df = pd.read_csv('RL/entropy.csv')
import warnings

# 在计算这些指标之前添加以下代码
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 定义强化学习环境
class CustomEnv(gym.Env):
    def __init__(self, ddos_df, entropy_df,window_size=20):
        super(CustomEnv, self).__init__()
        self.ddos_df = ddos_df
        self.entropy_df = entropy_df
        self.current_step = 0
        self.total_steps = len(entropy_df)/window_size
        self.window_size=window_size
        # 定义动作空间
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # 定义观察空间
        self.observation_space = spaces.Dict({
            'threshold': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'accuracy': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'dr': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'pr': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # 'fpr': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'f1': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # 'entropy_window': spaces.Box(low=0, high=np.inf, shape=(window_size,), dtype=np.float32),
            # 'ddos_window': spaces.Box(low=0, high=np.inf, shape=(window_size,), dtype=np.float32),
        })
    def reset(self):
        # 1. 重置环境状态，初始化当前步数等
        self.current_step = 0

        # 2. 读取初始窗口的数据
        # initial_window_entropy = self.read_next_window(self.entropy_df,self.window_size,(self.current_step*self.window_size))
        # initial_window_ddos = self.read_next_window(self.ddos_df,self.window_size,(self.current_step*self.window_size))
        # 3. 返回初始观察
        initial_observation = {
            'threshold': np.random.uniform(low=0, high=1, size=(1,)),
            'accuracy': np.random.uniform(low=0, high=1, size=(1,)),
            'dr': np.random.uniform(low=0, high=1, size=(1,)),
            'pr': np.random.uniform(low=0, high=1, size=(1,)),
            # 'fpr': np.random.uniform(low=0, high=1, size=(1,)),
            'f1': np.random.uniform(low=0, high=1, size=(1,)),
            # 'entropy_window': initial_window_entropy['normalized_entropy'].values,
            # 'ddos_window': initial_window_ddos['DDOS'].values,
        }

        return initial_observation


    def calculatemetrix(self, entropy_window, ddos_window):
        # 计算混淆矩阵
        # conf_matrix = confusion_matrix(entropy_window, ddos_window)
        # print(conf_matrix.shape)

        # 计算正确率
        accuracy = accuracy_score(ddos_window, entropy_window)
        # 计算精确率
        precision = precision_score(ddos_window, entropy_window)
        # 计算召回率
        recall = recall_score(ddos_window, entropy_window)        
        # 计算假正率
        # fpr = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[1, 0])
        # 计算 F1 分数
        f1 = f1_score(ddos_window, entropy_window)
        # 在计算这些指标后
        # print("Metrics:")
        # print(f"Accuracy: {accuracy:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F1 Score: {f1:.4f}")

        # print(accuracy,precision,recall,f1)
        return accuracy,recall,precision,f1
    def read_next_window(self,data_frame, window_size, current_index):
        end_index = current_index + window_size
        if end_index <= len(data_frame):
            window_data = data_frame.iloc[current_index:end_index]
        else:
            # Handle the case where the window extends beyond the end of the DataFrame
            # You might want to implement a specific behavior based on your requirements
            window_data = data_frame.iloc[current_index:]

        return window_data
    def calculate_reward(self, accuracy, recall, precision, f1):
        alpha = 0.25  # 调整为适当的值
        beta = 0.3    # 调整为适当的值
        gamma = 0.2   # 调整为适当的值
        delta = -0.25  # 调整为适当的值

        reward = alpha * accuracy + beta * recall + gamma * precision + delta * f1
        return reward
    def step(self, action):
        # Assuming 'action' is the selected threshold value from the action space

        # Read the next window of data from the CSV file
        # Assuming 'read_next_window' is a function that reads the next window of data
        next_window_entropy = self.read_next_window(self.entropy_df,self.window_size,self.current_step*self.window_size)
        next_window_ddos = self.read_next_window(self.ddos_df,self.window_size,self.current_step*self.window_size)

        # Compare entropy with threshold and update entropy_window
        entropy_window = (next_window_entropy['normalized_entropy'].values < action).astype(int)

        # Update ddos_window
        ddos_window = next_window_ddos['DDOS'].values

        # Calculate accuracy, dr, pr (assuming you have functions for these calculations)
        accuracy,recall,precision,f1 = self.calculatemetrix(ddos_window, entropy_window)
        
        # Update observation space
        observation = {
            'threshold': np.array([action]),
            'accuracy': np.array([accuracy]),
            'dr': np.array([recall]),
            'pr': np.array([precision]),
            # 'fpr': np.array([fpr]),
            'f1': np.array([f1]),
            'entropy_window': entropy_window,
            'ddos_window': ddos_window,
        }
        reward = self.calculate_reward(accuracy,recall,precision,f1)
        # 更新当前步数
        self.current_step += 1
        
        # 是否达到最大步数
        done = self.current_step >= self.total_steps
        if done:
            print("Optimal Threshold:",action)
            # 在计算这些指标后
            print("Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

        
        # Return observation, reward, done, and additional info
        return observation, reward, done, {}

# 创建强化学习环境
env = DummyVecEnv([lambda: CustomEnv(ddos_df, entropy_df)])

# 创建PPO模型
model = PPO("MultiInputPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 打印最优阈值和训练时间
# print("Optimal Threshold:", model.)
# print("Training Time:", model.duration)
