"""
MADDPG Agent Object
@author: udacity, KathleenWang
Created on 4/7/19
"""       
# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list, convert_to_tensor
from collections import deque, namedtuple
import random
import numpy as np


device = 'cpu'



class MADDPG:
    def __init__(self, in_a, h_in_a, h_o_a, o_a, in_c_s, in_c_a, h_in_c, h_o_c, seedn,  lr_actor, lr_critic, discount_factor, tau):
        super(MADDPG, self).__init__()

        # Each DDPGAgent contains an actor and a critic. Here we have two actors, hence 2 agents
        # network inputs can be defined during training
        
        self.maddpg_agent = [DDPGAgent(in_a, h_in_a, h_o_a, o_a, in_c_s, in_c_a, h_in_c, h_o_c,  lr_actor, lr_critic, seedn), 
                             DDPGAgent(in_a, h_in_a, h_o_a, o_a, in_c_s, in_c_a, h_in_c, h_o_c,  lr_actor, lr_critic, seedn)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.action_size = o_a


    def act(self, both_states, noise=0.0):
        """get actions from all agents in the MADDPG object"""               
        actions =  [self.maddpg_agent[idx].act(both_states[idx], noise)  for idx in range(len(self.maddpg_agent))]            
        return actions    
    
    
    def target_act(self, both_states, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """        
        target_actions =  [self.maddpg_agent[idx].target_act(both_states[idx], noise)  for idx in range(len(self.maddpg_agent))]        
        return target_actions

    
    def reset(self):
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()
    
    def transform_samples(self, samples):
        """The MADDPG agent takes in observed sample, find predicted actions and target actions from both agents
        and return them as output"""
        states, full_state, actions, rewards, next_states, next_full_state, dones = map(transpose_to_tensor, samples)        
        full_states = [samples[1], samples[5]]
        samples = [states, actions, rewards, next_states, dones]
        samples.extend(convert_to_tensor(full_states))   
        
        states = torch.stack(states)   
        next_states = torch.stack(next_states)       
        target_actions = self.target_act(next_states, noise = 0.0) 
        pred_actions = self.act(states, noise = 0.0 )
        return [samples, pred_actions, target_actions]
        
            
    def update(self, samplesa, samplesb):
        """update the critics and actors of all the agents 
       each sample has batch number of environment, each agent gets their own sample each time"""

        agents_inputs = []
        # agent a uses samplesa , agentb uses sampleb
        agents_inputs.append(self.transform_samples(samplesa))
        agents_inputs.append(self.transform_samples(samplesb))
        
        for i, inputs in enumerate(zip(self.maddpg_agent, agents_inputs)):
            ddpg_agent, agent_input = inputs
            samples = agent_input[0]
            pred_actions = agent_input[1]
            target_actions = agent_input[2]
            
            ddpg_agent.learn(samples, pred_actions, target_actions, agent_num=i , discount_factor=self.discount_factor)
            
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau) 
    



class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,transition):
        """push into the buffer"""
        self.deque.append(transition)

        
        
    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)



