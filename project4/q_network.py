import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_

class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        #--------- YOUR CODE HERE --------------
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 9)

        #---------------------------------------
        self.action = {}
        action_pool = np.array([-0.8,0,0.8])
        index=0 
        for i in range(len(action_pool)):
            for j in range(len(action_pool)):
                self.action[index]=np.array([action_pool[i],action_pool[j]])
                index+=1
       
        
    def forward(self, x, device):
        x=torch.as_tensor(x).float()
        #--------- YOUR CODE HERE --------------
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        #---------------------------------------
        
    def select_discrete_action(self, obs, device):
        # Put the observation through the network to estimate q values for all possible discrete actions
        
        est_q_vals = self.forward(obs.reshape((1,) + obs.shape), device)
        # Choose the discrete action with the highest estimated q value
        discrete_action = torch.argmax(est_q_vals, dim=1).tolist()[0]
        return discrete_action
                
    def action_discrete_to_continuous(self, discrete_action):
        #--------- YOUR CODE HERE --------------
        continuous_action = self.action[discrete_action]
        return continuous_action
        #---------------------------------------
