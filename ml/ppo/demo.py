import gym
from gym import spaces
from sklearn.base import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

# 读取CSV文件
ddos_df = pd.read_csv('ddos.csv')
entropy_df = pd.read_csv('entropy.csv')

# 定义强化学习环境
class CustomEnv(gym.Env):
    def __init__(self, ddos_df, entropy_df,window_size=10):
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
            'fpr': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'f1': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'entropy_window': spaces.Box(low=0, high=np.inf, shape=(window_size,), dtype=np.float32),
            'ddos_window': spaces.Box(low=0, high=np.inf, shape=(window_size,), dtype=np.float32),
        })

    def reset(self):
        self.current_step = 0
        # Initialize 'entropy_window' as an empty array
        entropy_window = np.empty(self.window_size)

        # Initialize 'ddos_window' as an array containing -1
        ddos_window = np.full((self.window_size,), -1)

        # Your reset logic goes here
        # For example, you can initialize other components randomly
        initial_state = {
            'threshold': np.random.uniform(low=0, high=1, size=(1,)),
            'accuracy': np.random.uniform(low=0, high=1, size=(1,)),
            'dr': np.random.uniform(low=0, high=1, size=(1,)),
            'pr': np.random.uniform(low=0, high=1, size=(1,)),
            'fpr': np.random.uniform(low=0, high=1, size=(1,)),
            'f1': np.random.uniform(low=0, high=1, size=(1,)),
            'entropy_window': entropy_window,
            'ddos_window': ddos_window,
        }
        return initial_state
    def calculate_f1(true_positive, false_positive, false_negative):
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1
    def calculatemetrix(self, entropy_window, ddos_window):
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(entropy_window, ddos_window)
        tn, fp, fn, tp = np.array(conf_matrix).ravel()

        # 输出混淆矩阵
        # print("Confusion Matrix:")
        # print(conf_matrix)
        # 计算正确率
        accuracy = accuracy_score(entropy_window, ddos_window)
        dr=tp/(tp+fn)
        fpr=fp/(fp+tn)
        pr=tp/(tp+fp)
        f1=self.calculate_f1(tp,fp,fn)
        # print(accuracy,dr,fpr,pr)
        return accuracy,dr,fpr,pr,f1
    def read_next_window(data_frame, window_size, current_index):
        end_index = current_index + window_size
        if end_index <= len(data_frame):
            window_data = data_frame.iloc[current_index:end_index]
        else:
            # Handle the case where the window extends beyond the end of the DataFrame
            # You might want to implement a specific behavior based on your requirements
            window_data = data_frame.iloc[current_index:]

        return window_data
    def step(self, action):
        # Assuming 'action' is the selected threshold value from the action space

        # Read the next window of data from the CSV file
        # Assuming 'read_next_window' is a function that reads the next window of data
        next_window_entropy = self.read_next_window(self.entropy_df,self.window_size,self.current_step*self.window_size)
        next_window_ddos = self.read_next_window(self.ddos_df,self.window_size,self.current_step*self.window_size)

        # Compare entropy with threshold and update entropy_window
        entropy_window = (next_window_entropy['entropy'].values < action).astype(int)

        # Update ddos_window
        ddos_window = next_window_ddos['ddos'].values

        # Calculate accuracy, dr, pr (assuming you have functions for these calculations)
        accuracy,dr,fpr,pr,f1 = self.calculatemetrix(ddos_window, entropy_window)
        
        # Update observation space
        observation = {
            'threshold': np.array([action]),
            'accuracy': np.array([accuracy]),
            'dr': np.array([dr]),
            'pr': np.array([pr]),
            'fpr': np.array([fpr]),
            'f1': np.array([f1]),
            'entropy_window': entropy_window,
            'ddos_window': ddos_window,
        }
        reward = 0.5 * accuracy + 0.3 * dr + 0.2 * pr - 0.1 * fpr - 0.4 * (1 - f1)
        # 更新当前步数
        self.current_step += 1
        
        # 是否达到最大步数
        done = self.current_step >= self.total_steps-1
        if done:
            print("Optimal Threshold:",action)
        
        # Return observation, reward, done, and additional info
        return observation, reward, done, {}

# 创建强化学习环境
env = DummyVecEnv([lambda: CustomEnv(ddos_df, entropy_df)])

# 创建PPO模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 打印最优阈值和训练时间
# print("Optimal Threshold:", model.)
# print("Training Time:", model.duration)
