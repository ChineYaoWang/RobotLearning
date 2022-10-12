import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import argparse
import os
import math
from tqdm import tqdm

np.set_printoptions(suppress=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)
    parser.add_argument('--time_limit', type=float, default=5)
    parser.add_argument('--save_dir', type=str, default='dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    arm_teacher = Robot(dynamics_teacher)

    # ---
    # You code goes here. Replace the X, and Y by your collected data
    # Control the arm to collect a dataset for training the forward dynamics.
    if args.num_links ==1:
        num_samples=20000
    elif args.num_links ==2:
        num_samples=100000
    else:
        num_samples=300000
    X = np.zeros((arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim(), num_samples))
    Y = np.zeros((arm_teacher.dynamics.get_state_dim(), num_samples))
    #parametes
    state=np.zeros((arm_teacher.dynamics.get_state_dim(), 1))
    action=np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
    action_set =[-0.08,-0.03,-0.01,0,0.01,0.03,0.08]
    #initial
    state[0]=-90*math.pi/180
    arm_teacher.set_state(state)
    arm_teacher.set_action(action)


    if args.num_links ==1:
        for i in range(num_samples):
            j = 0
            while j < len(action):
                index=np.random.randint(len(action_set))
                action[j]=action[j]+action_set[index]
                if abs(action[j])<1.5:
                    j+=1
                else:
                    action[j] -= action_set[index]
            #store X        
            X[0:2,i]=state[:, 0]
            X[2:3,i]=action[:, 0]
            #set new action
            arm_teacher.set_action(action)
            #apply
            arm_teacher.advance()
            #new state
            state=arm_teacher.get_state()

            Y[:,i:i+1]=state
    if args.num_links ==2:
        for i in tqdm(range(num_samples), total=num_samples):
            #store X        
            X[0:4,i]=state[:, 0]
            X[4:6,i]=action[:, 0]
            j = 0
            while j < len(action):
                index=np.random.randint(len(action_set))
                action[j]=action[j]+action_set[index]
                predict_state=dynamics_teacher.advance(state,action)
                predict_pos_ee=dynamics_teacher.compute_fk(predict_state)
                if abs(action[j])<1.5:
                    j+=1
                else:
                    action[j] -= action_set[index]
                
                if predict_pos_ee[1] >=0:
                    state[0]=-90*math.pi/180
                    state[1:] = 0
                    action[:] = 0
                    j = 0
                    arm_teacher.set_state(state)
                    arm_teacher.set_action(action)
                
            #set new action
            arm_teacher.set_action(action)
            #apply
            arm_teacher.advance()
            #new state
            state=arm_teacher.get_state()
            Y[:,i:i+1]=state
            
    if args.num_links ==3:
        for i in tqdm(range(num_samples), total=num_samples):
            #store X        
            X[0:6,i]=state[:, 0]
            X[6:9,i]=action[:, 0]
            j = 0
            while j < len(action):
                index=np.random.randint(len(action_set))
                action[j]=action[j]+action_set[index]
                predict_state=dynamics_teacher.advance(state,action)
                predict_pos_ee=dynamics_teacher.compute_fk(predict_state)
                if abs(action[j])<1.5:
                    j+=1
                else:
                    action[j] -= action_set[index]
                
                if predict_pos_ee[1] >=0:
                    state[0]=-90*math.pi/180
                    state[1:] = 0
                    action[:] = 0
                    j = 0
                    arm_teacher.set_state(state)
                    arm_teacher.set_action(action)
                
            #set new action
            arm_teacher.set_action(action)
            #apply
            arm_teacher.advance()
            #new state
            state=arm_teacher.get_state()
            Y[:,i:i+1]=state
   
    # ---

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(os.path.join(args.save_dir, 'X.npy'), X)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y)
