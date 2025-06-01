import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import numpy as np
from gymnasium import spaces

class MarioEnv(gym.Env):
    """超级马里奥游戏环境"""
    def __init__(self):
        super().__init__()
        
        # 定义观察空间 (84x84x3 RGB图像)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        
        # 定义动作空间 (不动, 左, 右, 跳, 跑, 左跳, 右跳, 左跑, 右跑)
        self.action_space = spaces.Discrete(9)
        
        # 游戏状态
        self.state = None
        self.score = 0
        self.steps = 0
        self.max_steps = 200  # 减少每个回合的最大步数
        self.positions = []   # 记录马里奥的位置
        self.last_x = 0      # 上一次的x位置
        
    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        self.state = np.zeros((84, 84, 3), dtype=np.uint8)
        self.score = 0
        self.steps = 0
        self.positions = []
        self.last_x = 40  # 初始x位置
        
        # 初始化马里奥的起始位置
        self._update_state()
        
        return self.state, {}
        
    def step(self, action):
        """执行一个动作并返回新的状态"""
        self.steps += 1
        
        # 根据动作更新马里奥的位置
        new_x = self.last_x
        if action in [1, 5, 7]:  # 左移动作
            new_x = max(0, self.last_x - 2)
        elif action in [2, 6, 8]:  # 右移动作
            new_x = min(80, self.last_x + 2)
        
        self.last_x = new_x
        self.positions.append(new_x)
        
        # 计算奖励
        reward = self._calculate_reward(new_x)
        
        # 更新状态
        self._update_state()
        
        # 检查是否结束
        done = self.steps >= self.max_steps
        truncated = False
        
        info = {
            'x_pos': new_x,
            'score': self.score,
            'steps': self.steps
        }
        
        return self.state, reward, done, truncated, info
    
    def _update_state(self):
        """更新游戏状态（图像）"""
        # 创建一个简单的RGB图像表示
        self.state = np.zeros((84, 84, 3), dtype=np.uint8)
        # 在马里奥的位置画一个红色方块
        x = int(self.last_x)
        self.state[70:80, x:x+10, 0] = 255  # Red channel
        # 添加一些简单的背景
        self.state[80:, :, 1] = 100  # 绿色地面
        self.state[0:20, :, 2] = 200  # 蓝色天空
        
    def _calculate_reward(self, new_x):
        """计算奖励"""
        reward = 0
        
        # 向右移动的主要奖励
        movement_reward = new_x - self.last_x
        reward += movement_reward * 0.5  # 增加向右移动的奖励
        
        # 达到新的最远距离的额外奖励
        if new_x > max(self.positions, default=0):
            reward += 2.0
        
        # 停留惩罚
        if len(self.positions) > 5:
            recent_positions = self.positions[-5:]
            if max(recent_positions) - min(recent_positions) < 3:
                reward -= 1.0
        
        # 保持在安全区域的小奖励
        if 20 <= new_x <= 60:
            reward += 0.1
        
        # 完成关卡奖励
        if new_x >= 80:
            reward += 100.0
        
        return reward

def make_env():
    env = MarioEnv()
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    return env

def main():
    # 创建环境
    env = make_vec_env(
        make_env,
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    
    # 创建模型，使用更简单的MLP策略
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.001,    # 提高学习率
        n_steps=512,           # 减少每次更新的步数
        batch_size=32,         # 减少批量大小
        n_epochs=5,            # 减少训练轮次
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log="./mario_tensorboard/"
    )
    
    # 训练模型（大幅减少训练步数）
    model.learn(total_timesteps=10000)   # 减少到1万步
    
    # 保存模型
    model.save("mario_model")

if __name__ == "__main__":
    main()
