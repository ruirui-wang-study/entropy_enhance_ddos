import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

# 读取CSV文件
ddos_df = pd.read_csv('ddos.csv')
entropy_df = pd.read_csv('entropy.csv')

# 定义强化学习环境
class CustomEnv(gym.Env):
    def __init__(self, ddos_df, entropy_df):
        super(CustomEnv, self).__init__()
        self.ddos_df = ddos_df
        self.entropy_df = entropy_df
        self.current_step = 0
        self.total_steps = len(entropy_df)
        
        # 定义动作空间
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # 定义观测空间
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return np.array([self.entropy_df['normalized_entropy'].iloc[self.current_step]])

    def step(self, action):
        # 计算准确率
        threshold = action[0]
        
        predicted_ddos = (self.entropy_df['normalized_entropy'] < threshold).astype(int)
        accuracy = np.mean(predicted_ddos == self.ddos_df['DDOS'].values)
        
        # 制定reward
        reward = accuracy
        
        # 更新当前步数
        self.current_step += 1
        
        # 是否达到最大步数
        done = self.current_step >= self.total_steps-1
        if done:
            print("Optimal Threshold:",threshold)
        
        # 返回观测、reward、是否结束、额外信息
        return np.array([self.entropy_df['normalized_entropy'].iloc[self.current_step]]), reward, done, {}

# 创建强化学习环境
env = DummyVecEnv([lambda: CustomEnv(ddos_df, entropy_df)])

# 创建PPO模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 打印最优阈值和训练时间
# print("Optimal Threshold:", model.)
# print("Training Time:", model.duration)
