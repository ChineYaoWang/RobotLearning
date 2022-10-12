import sys
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from arm_dynamics_student import ArmDynamicsStudent
from robot import Robot
from arm_gui import ArmGUI, Renderer
import argparse
import time
import math
import torch
import random
np.set_printoptions(suppress=True)

class MPC:

    def __init__(self):
        self.control_horizon = 10
        # Define other parameters here
        self.min_loss=100000;
        self.new_state=[0,0,0,0,0,0];   #important
        
        self.delta_action=[-0.04,0.04]
        #self.delta_action1=[-0.02,0.02]
        '''
        self.delta_action1=[-0.1,0.1]
        self.delta_action2=[-0.05,0.05]
        self.delta_action3=[-0.015,0.015]
        '''
  
        
        
        self.delta_action1=[-0.08,-0.03,-0.01,0,0.01,0.03,0.08]
        self.delta_action2=[-0.08,-0.04,-0.01,0,0.01,0.04,0.08]
        self.delta_action3=[-0.08,-0.04,-0.01,0,0.01,0.04,0.08]
        
        self.brake1=[-0.01,0,0.01]
        self.brake2=[-0.02,0,0.02]
        self.brake3=[-0.02,0,0.02]
        
        self.best_action=np.zeros((3,1))
        self.s=[self.delta_action1,self.delta_action2,self.delta_action3]
     
    def calculate_loss_N(self,dynamics,state,goal,action,N):
        # calculate n - state position
        sum=0
        sum_a=0
        if dynamics.get_action_dim() == 3:
            new_state_pos=dynamics.compute_fk(state).copy()
        for i in range(N):
            state=dynamics.dynamics_step(state,action,dynamics.dt)
            #state=self.new_state
            new_state_pos=dynamics.compute_fk(state).copy()
            
            sum=sum+np.sqrt(pow(goal[0]-new_state_pos[0],2)+pow(goal[1]-new_state_pos[1],2))
        v_cost=np.sqrt(pow(dynamics.compute_vel_ee(state)[0],2)+pow(dynamics.compute_vel_ee(state)[1],2))
        if dynamics.get_action_dim() <=2:
            sum=np.sqrt(pow(goal[0]-new_state_pos[0],2)+pow(goal[1]-new_state_pos[1],2))
            a=1
            b=0.03
            c=0
        else:
            a=sum
            b=v_cost*14
            c=6
            sum_a=action[0]*action[0]+action[1]*action[1]+action[2]*action[2]
      
              
    
       
        
        return  a*sum+b*v_cost+c*sum_a
        
    def compute_action(self, dynamics, state, goal, action):
        # Put your code here. You must return an array of shape (num_links, 1)
        
        ct = np.zeros((dynamics.get_action_dim(),1))  
        self.min = self.calculate_loss_N(dynamics, state, goal, action,30)
        
        if(dynamics.get_action_dim()==3): 
            current_pos=np.sqrt(pow(goal[0]-dynamics.compute_fk(state)[0],2)+pow(goal[1]-dynamics.compute_fk(state)[1],2))
            current_V=np.sqrt(pow(dynamics.compute_vel_ee(state)[0],2)+pow(dynamics.compute_vel_ee(state)[1],2))
            #link1
            for i in self.delta_action1:
                current_act1 = action[0]+i
             
                ct[0] = current_act1
                ct[1] = action[1]
                ct[2] = action[2]
                loss = self.calculate_loss_N(dynamics, state, goal, ct,30)
  
                if(loss<self.min_loss):
                    self.min_loss = loss
                    self.best_action = np.copy(ct)
                  
            #link2  
            for j in self.delta_action2:
                current_act2 = action[1]+j
                ct[0] = self.best_action[0]
                ct[1] = current_act2
                ct[2] = action[2]
                loss = self.calculate_loss_N(dynamics, state, goal, ct,30)
                if(loss<self.min_loss):
                    self.min_loss = loss
                    self.best_action = np.copy(ct)
            #link3
            for k in self.delta_action3:
                current_act3 = action[2]+k
                ct[0] = self.best_action[0]
                ct[1] = self.best_action[1]
                ct[2] = current_act3
                loss = self.calculate_loss_N(dynamics, state, goal, ct,30)
                if(loss<self.min_loss):
                    self.min_loss = loss
                    self.best_action = np.copy(ct)

        if(dynamics.get_action_dim()==2):
            for i in self.delta_action:
                current_act1 = action[0]+i
                for j in self.delta_action:
                    current_act2 = action[1]+j
                    ct[0] = current_act1
                    ct[1] = current_act2
                    loss = self.calculate_loss_N(dynamics, state, goal, ct,30)
                    if(loss<self.min_loss):
                        self.min_loss = loss
                        self.best_action = np.copy(ct)
        
        if dynamics.get_action_dim()==1 :
            for i in self.delta_action:
                current_act1 = action[0]+i
                ct[0] = current_act1
                loss = self.calculate_loss_N(dynamics, state, goal, ct,30)
                if loss < self.min_loss:
                    self.min_loss = loss
                    self.best_action = np.copy(ct)
      
        
        return self.best_action
        # Don't forget to comment out the line below
        #raise NotImplementedError("MPC not implemented")



def main(args):

    # Arm
    arm = Robot(
        ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
    )

    # Dynamics model used for control
    if args.model_path is not None:
        dynamics = ArmDynamicsStudent(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
        dynamics.init_model(args.model_path, args.num_links, args.time_step, device=torch.device("cpu"))
    else:
        # Perfectly accurate model dynamics
        dynamics = ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )

    # Controller
    controller = MPC()

    # Control loop
    arm.reset()
    action = np.zeros((arm.dynamics.get_action_dim(), 1))
    goal = np.zeros((2, 1))
    goal[0, 0] = args.xgoal
    goal[1, 0] = args.ygoal
    arm.goal = goal

    if args.gui:
        renderer = Renderer()
        time.sleep(0.25)

    dt = args.time_step
    k = 0
    
    while True:
        t = time.time()
        arm.advance()
        if args.gui:
            renderer.plot([(arm, "tab:blue")])
        k += 1
        time.sleep(max(0, dt - (time.time() - t)))
        if k == controller.control_horizon:
            state = arm.get_state()
            action = controller.compute_action(dynamics, state, goal, action)
            arm.set_action(action)
            #print(action)
            k = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_links", type=int, default=3)
    parser.add_argument("--link_mass", type=float, default=0.1)
    parser.add_argument("--link_length", type=float, default=1)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--time_limit", type=float, default=5)
    parser.add_argument("--gui", action="store_const", const=True, default=False)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--xgoal", type=float, default=-1.19286783)
    parser.add_argument("--ygoal", type=float, default=-1.28045552)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
