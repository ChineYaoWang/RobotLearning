from arm_dynamics_base import ArmDynamicsBase
import numpy as np
import torch
from models import build_model


class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, time_step, device):
        # ---
        # Your code hoes here
        # Initialize the model loading the saved model from provided model_path
        self.model = build_model(num_links,time_step)
        self.model.load_state_dict(torch.load(model_path))
        self.X=torch.zeros((1, num_links*3))
        # ---
        if self.model:
            self.model_loaded = True

    def dynamics_step(self, state, action, dt):
        if self.model_loaded:
            # ---
            # Your code goes here
            # Use the loaded model to predict new state given the current state and action
            self.model.eval()
            state = torch.transpose(torch.FloatTensor(state), 0, 1)
            action = torch.transpose(torch.FloatTensor(action), 0, 1)
            length = state.size()[1]
            self.X[0:1, 0:length]=state
            self.X[0:1, length:length+action.size()[1]] = action
            new_state = self.model(self.X)
            new_state = torch.transpose(new_state, 0, 1)
            return new_state.detach().numpy()
            # ---
        else:
            return state

