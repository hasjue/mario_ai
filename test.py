import gymnasium as gym
import pygame
import numpy as np
from stable_baselines3 import PPO
from train import MarioEnv, make_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def initialize_pygame(width=840, height=840):
    """初始化Pygame显示"""
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Super Mario AI")
    return screen

def display_state(screen, state, reward, step):
    """显示游戏状态"""
    # 清空屏幕
    screen.fill((0, 0, 0))
    
    # 将状态图像放大显示
    state_surface = pygame.surfarray.make_surface(np.squeeze(state))
    state_surface = pygame.transform.scale(state_surface, (840, 840))
    screen.blit(state_surface, (0, 0))
    
    # 显示信息
    font = pygame.font.Font(None, 36)
    reward_text = font.render(f"Reward: {reward:.2f}", True, (255, 255, 255))
    step_text = font.render(f"Step: {step}", True, (255, 255, 255))
    screen.blit(reward_text, (10, 10))
    screen.blit(step_text, (10, 50))
    
    pygame.display.flip()

def main():
    # 创建环境
    env = make_env()
    
    # 加载训练好的模型
    try:
        model = PPO.load("mario_model")
        print("加载已训练的模型")
    except:
        print("找不到训练好的模型，使用随机动作")
        model = None
    
    # 初始化显示
    screen = initialize_pygame()
    
    # 测试循环
    obs, _ = env.reset()
    total_reward = 0
    step = 0
    
    running = True
    while running:
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 获取动作
        if model:
            action, _states = model.predict(obs)
        else:
            action = env.action_space.sample()
        
        # 执行动作
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # 显示状态
        display_state(screen, obs, total_reward, step)
        
        # 控制帧率
        pygame.time.wait(50)
        
        if done or truncated:
            print(f"Episode finished. Total reward: {total_reward}")
            obs, _ = env.reset()
            total_reward = 0
            step = 0
    
    pygame.quit()

if __name__ == "__main__":
    main()
