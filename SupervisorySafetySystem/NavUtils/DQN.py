import gym
import collections
import random
import sys
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.996



class SmartBufferDQN(object):
    def __init__(self, max_size=1000000, state_dim=14):     
        self.max_size = max_size
        self.state_dim = state_dim
        self.ptr = 0

        self.states = np.empty((max_size, state_dim))
        self.actions = np.empty((max_size, 1))
        self.next_states = np.empty((max_size, state_dim))
        self.rewards = np.empty((max_size, 1))
        self.dones = np.empty((max_size, 1))

    def add(self, s, a, s_p, r, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.next_states[self.ptr] = s_p
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = d

        self.ptr += 1
        
        if self.ptr == 99999: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, 1))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=MEMORY_SIZE)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, obs_space, action_space, h_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
class DQN:
    def __init__(self, obs_space, action_space, name):
        self.name = name
        self.obs_space = obs_space
        self.action_space = action_space

        self.model = None
        self.target = None
        self.optimizer = None

        # self.memory = ReplayBuffer()
        self.replay_buffer = SmartBufferDQN(state_dim=obs_space)
        self.exploration_rate = EXPLORATION_MAX
        self.update_steps = 0

    def create_agent(self, h_size):
        self.model = Qnet(self.obs_space, self.action_space, h_size)
        self.target = Qnet(self.obs_space, self.action_space, h_size)
        self.target.load_state_dict(self.model.state_dict())


    def act(self, obs):
        if random.random() < self.exploration_rate:
            return random.randint(0,self.action_space-1)
        else: 
            return self.greedy_action(obs)

    def greedy_action(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        return out.argmax().item()


    def train(self, n_train=1):
        for i in range(n_train):
            if self.replay_buffer.size() < BATCH_SIZE:
                return
            s, a, s_p, r, done = self.replay_buffer.sample(BATCH_SIZE)
            s = torch.from_numpy(s).float()
            a = torch.LongTensor(a)
            s_p = torch.from_numpy(s_p).float()
            r = torch.from_numpy(r).float()
            done = torch.from_numpy(done).float()

            # obs_t_p = torch.from_numpy(s_p).float()
            next_values = self.target.forward(s_p)
            max_vals = torch.max(next_values, dim=1)[0].reshape((BATCH_SIZE, 1))
            g = torch.ones_like(done) * GAMMA
            q_update = r + g * max_vals * done
            q_vals = self.model.forward(s)
            q_a = q_vals.gather(1, a)
            loss = F.mse_loss(q_a, q_update.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_networks()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 12 == 1:
            self.exploration_rate *= EXPLORATION_DECAY 
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save(self, directory="./saves"):
        filename = self.name

        torch.save(self.model, '%s/%s_model.pth' % (directory, filename))
        torch.save(self.target, '%s/%s_target.pth' % (directory, filename))

    def load(self, directory="./saves"):
        filename = self.name
        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
        self.target = torch.load('%s/%s_target.pth' % (directory, filename))

        print("Agent Loaded")

    def try_load(self, load=True, h_size=100, path=None):
        if load:
            try:
                self.load(path)
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Unable to load model")
                pass
        else:
            print(f"Not loading - restarting training")
            self.create_agent(h_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)



def observe(env, memory, n_itterations=10000):
    s = env.reset()
    done = False
    for i in range(n_itterations):
        action = env.action_space.sample()
        s_p, r, done, _ = env.step(action)
        done_mask = 0.0 if done else 1.0
        memory.add(s, action, s_p, r/100, done_mask)
        s = s_p
        if done:
            s = env.reset()

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()

def test_cartpole():
    env = gym.make('CartPole-v1')
    agent = DQN(env.observation_space.shape[0], env.action_space.n, "CartpoleAgent")
    agent.create_agent(24)

    print_n = 20

    rewards = []
    observe(env, agent.replay_buffer)
    for n in range(500):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.act(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            agent.replay_buffer.add(state, a, s_prime, r/100, done_mask)
            # agent.memory.put((state, a, r/100, s_prime, done_mask))
            state = s_prime
            score += r
            agent.train()
            
        rewards.append(score)
        if n % print_n == 1:
            print(f"Run: {n} --> Score: {score} --> Mean: {np.mean(rewards[-20:])} --> exp: {agent.exploration_rate}")



if __name__ == '__main__':
    test_cartpole()

