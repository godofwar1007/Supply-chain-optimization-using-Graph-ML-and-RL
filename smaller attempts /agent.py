import numpy as np 
import random
import torch 
import torch.nn as nn 
import torch.optim as optim 
from collections import deque 
import gymnasium as gym 


class Qnetwork(nn.Module):

    def __init__(self,obs_dim,action_dim,hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim)
        )

    def forward(self,x):
        return self.net(x)

class Replay_Buffer():

    def __init__(self,capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self,obs,action,reward,next_obs,done):
        self.buffer.append((obs,action,reward,next_obs,done))

    def sample(self, batch_size):

        batch  = random.sample(self.buffer,batch_size)
        obs_batch = torch.FloatTensor([b[0] for b in batch])
        action_batch = torch.LongTensor([b[1] for b in batch])
        reward_batch = torch.FloatTensor([b[2] for b in batch])
        next_obs_batch = torch.FloatTensor([b[3] for b in batch])
        done_batch = torch.FloatTensor([b[4] for b in batch])

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch               

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim, action_dim, 
                 learning_rate=1e-3, 
                 gamma=0.99, 
                 epsilon_start=1.0, 
                 epsilon_end=0.01, 
                 epsilon_decay=0.995,
                 buffer_capacity=10000,
                 batch_size=64,
                 target_update_freq=100):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0
        
        # Q-networks
        self.q_network = Qnetwork(obs_dim, action_dim)
        self.target_network = Qnetwork(obs_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = Replay_Buffer(buffer_capacity)

    def act(self, obs, eval_mode=False):
        """Select action using epsilon-greedy policy."""
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            return torch.argmax(q_values).item()
    
    def remember(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        # Compute current Q values
        current_q = self.q_network(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network(next_obs_batch).max(1)[0]
            target_q = reward_batch + self.gamma * next_q * (1 - done_batch)
        
        # Loss and optimization
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon (decay)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']            
