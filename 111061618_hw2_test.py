# test
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F 
import random

import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

import time
import os

class Net(nn.Module):
    def __init__(self, s_size, a_size):  # 這邊self後面加上','是原本就這樣的
        super(Net, self).__init__()  # 提醒，super是固定項
        self.s_size = s_size
        self.a_size = a_size
        self.fc1 = nn.Linear(self.s_size, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(100, 1024)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(1024, 512)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(512, self.a_size)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        actions_value = self.out(x)
        return actions_value

class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)



        self.eval_net, self.target_net = Net(
            int(240*256/4), 12), Net(int(240*256/4), 12)
        

        # 參數區
        self.LR = 0.001
        self.EPSILON_end = 0.9
        self.EPSILON_init = 0.3
        self.EPSILON = 0.9
        self.EPSILON_inc = 0.002

        self.memory_capacity = 1000
        self.target_replace_iter = 100

        self.batch_size = 32
        self.gamma = 0.9
        self.counter = 0

        self.learn_step_counter = 0  # 跑幾回合
        self.memory_counter = 0  # 記憶體要存在哪個位置
        # 這邊要初始化記憶體，裡面要存s,a,r,s_
        self.memory = np.zeros(
            (self.memory_capacity, (self.eval_net.s_size)*2 + 2))
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

        

    def act(self, observation):
        
        # #這邊預設輸入進來的obs是最原始的 應該要幫他轉灰階 resize
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = cv2.resize(observation, (int(observation.shape[0]/2), int(observation.shape[1]/2)), interpolation=cv2.INTER_AREA)

        observation = torch.FloatTensor(observation.flatten())
        #input only one sample
        if np.random.uniform() < 1:#greedy
            action_value = self.eval_net.forward(observation) 
            action = torch.argmax(action_value).data.numpy()

        else: # random
            action = self.action_space.sample()

        return int(action)



    def store_transition(self, s, a, r, s_):  # 存到記憶體
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity  # 如果記憶體超過最大值 從頭開始覆蓋
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn_dqn(self):
        if self.memory_counter < self.memory_capacity:
            return
        
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(
                self.eval_net.state_dict())  # 把eval的值複製到target
        self.learn_step_counter += 1

        # sample batch transitions
        # 這邊會在MEMORY_CAPACITY中隨機抽取BATCH_SIZE個index
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]

        # 把存取到的資料分開來存
        b_s = torch.FloatTensor(b_memory[:, :self.eval_net.s_size])
        b_a = torch.LongTensor(
            b_memory[:, self.eval_net.s_size:self.eval_net.s_size+1].astype(int))
        b_r = torch.FloatTensor(
            b_memory[:, self.eval_net.s_size+1:self.eval_net.s_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.eval_net.s_size:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # shape(batch,1)

        q_next = self.target_net(b_s_).detach()  # 加上detach代表不會使target_net被更新
        # detach from graph, don't backpropagate

        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        #shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
agent = Agent()
agent.eval_net.load_state_dict(torch.load('111061618_hw2_data.pth'))

Episodes = 50

total_r = 0

for epi in tqdm(range(Episodes)):
    
    ter = False
    state = env.reset()
    #state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    #state = cv2.resize(state, (int(state.shape[0]/2), int(state.shape[1]/2)), interpolation=cv2.INTER_AREA)

    while not ter:
        action = agent.act(state)
        
        state_, reward, ter, info = env.step((action))
        #state_ = cv2.cvtColor(state_, cv2.COLOR_BGR2GRAY)
        #state_ = cv2.resize(state_, (int(state_.shape[0]/2), int(state_.shape[1]/2)), interpolation=cv2.INTER_AREA)
        print(f"{info} ter :{ter}", end="\r")

        total_r += reward
        state = state_
        
        

print(total_r/50)