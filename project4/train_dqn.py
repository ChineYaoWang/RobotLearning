import argparse
import numpy as np
import time
import random
import os
import signal
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
import time

from replay_buffer import ReplayBuffer
from q_network import QNetwork
from arm_env import ArmEnv
import random
from torch.optim.lr_scheduler import MultiStepLR
import math
from collections import namedtuple
import time

def set_seed(SEED=0):
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
#---------- Utils for setting time constraints -----#
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
#---------- End of timing utils -----------#


class TrainDQN:

    @staticmethod
    def add_arguments(parser):
        # Common arguments
        parser.add_argument('--learning_rate', type=float, default=7e-4,
                            help='the learning rate of the optimizer')
        # LEAVE AS DEFAULT THE SEED YOU WANT TO BE GRADED WITH
        parser.add_argument('--seed', type=int, default=1,
                            help='seed of the experiment')
        parser.add_argument('--save_dir', type=str, default='models',
                            help="the root folder for saving the checkpoints")
        parser.add_argument('--gui', action='store_true', default=False,
                            help="whether to turn on GUI or not")
        # 7 minutes by default
        parser.add_argument('--time_limit', type=int, default=7*60,
                            help='time limits for running the training in seconds')

    def __init__(self, env, device, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        self.env = env
        self.env.seed(args.seed)
        self.env.observation_space.seed(args.seed)
        self.device = device
        self.q_network = QNetwork(env).to(self.device)
        #print(self.device.__repr__())
        #print(self.q_network)
        
        max_storage = 5000
        self.store = ReplayBuffer(max_storage)
        self.target_q_network = QNetwork(env).to(self.device)
        self.epsillon = 0.1 #0.1
        self.samples_num = 60 #60
        self.discount = 0.999 #0.999
        
    def save_model(self, episode_num, episode_reward, args):
        model_folder_name = f'episode_{episode_num:06d}_reward_{round(episode_reward):03d}'
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(self.q_network.state_dict(), os.path.join(args.save_dir, model_folder_name, 'q_network.pth'))
        #print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "q_network.pth")}\n')
                
    def train(self, args):
        #--------- YOUR CODE HERE --------------
        EPiSODES = 248
        optimizer = torch.optim.Adam(self.q_network.parameters(), lr=7e-4) #7e-4
        #scheduler = MultiStepLR(optimizer, milestones=[350], gamma=0.1)
        criterion = nn.SmoothL1Loss()
        done = False
        self.target_q_network.eval()
        s=1
        total_reward=0
        past =time.time()
        for episode in range(1, EPiSODES+1):
            # new episode
            done = False 
            obs = self.env.reset()
            total_loss=0
            step =0

       
            while done == False :
                # Get action using epsillon greedy
                self.q_network.eval()
                with torch.no_grad():
                    disc_action = self.q_network.select_discrete_action(obs,self.device)
                
                if random.random() < self.epsillon:
                    disc_action = random.choice([x for x in range(9) if x != disc_action])
             
                
                cont_action = self.q_network.action_discrete_to_continuous(disc_action)
                new_obs, reward, done, _ = self.env.step(cont_action)
                # Check if episode end
                if done == True :
                    self.save_model(episode,reward,args)  
                else:
                    # Store the transition
                    
                    self.store.put([obs,disc_action,reward,new_obs,done])
                    
                                   
                    if s >= self.samples_num: s = self.samples_num
                    
                    # Samples
                    mini_batch = self.store.sample(s)
                    s+=1
                    total_reward+=reward 
                    #print(Transition(*zip(*mini_batch)))
                    
                    # Train
                    self.q_network.train()
                    act =torch.as_tensor(mini_batch[1].reshape(-1,1))
                    predict = self.q_network(mini_batch[0],self.device).gather(1,act)
  
               
                    #print(torch.max(self.target_q_network(mini_batch[3],self.device),dim=1).values.detach())
                    a,_ = torch.max(self.target_q_network(mini_batch[3],self.device),dim=1)
                    a.detach()
                    #print("old")
                    #print(torch.max(self.target_q_network(mini_batch[3],self.device),dim=1).values)
                    #print(torch.as_tensor(mini_batch[2],dtype=torch.float32)s)
                    labels = torch.add(torch.as_tensor(mini_batch[2],dtype=torch.float32) , self.discount*a).reshape(-1,1)
                    
                 
                    loss = criterion(predict, labels)
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    # Update observation
                    obs = new_obs
              
              
                 
            if episode % 5 == 0: self.target_q_network.load_state_dict(self.q_network.state_dict())
            
            if episode >250:
                self.epsillon = 0.05 #0.05
            
            #scheduler.step()
            print(episode,total_reward/episode,time.time()-past)
            
            #---------------------------------------        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    TrainDQN.add_arguments(parser)
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    if not args.seed: args.seed = int(time.time())

    env = ArmEnv(args)
    device = torch.device('cpu')
    # declare Q function network and set seed
    tdqn = TrainDQN(env, device, args)
    # run training under time limit
    try:
        with time_limit(args.time_limit):
            tdqn.train(args)
    except TimeoutException as e:
        print("You ran out of time and your training is stopped!")
    
