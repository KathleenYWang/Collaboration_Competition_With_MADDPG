# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list 
from collections import deque, namedtuple
import random
import numpy as np


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    def __init__(self, in_a, h_in_a, h_o_a, o_a, in_c, h_in_c, h_o_c, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        # DDPGAgent: in_actor (input dim), hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic
        # Each DDPGAgent contains an actor and a critic. Here we have two actors, hence 2 agents
        
        self.maddpg_agent = [DDPGAgent(in_a, h_in_a, h_o_a, o_a, in_c, h_in_c, h_o_c), 
                             DDPGAgent(in_a, h_in_a, h_o_a, o_a, in_c, h_in_c, h_o_c)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, both_states, noise=0.0):
        """get actions from all agents in the MADDPG object"""

        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, both_states)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents 
       each sample has batch number of environment, each environment contains a state that can go into either agent"""

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        # previously, the obs contains

    
        obs_full_org, action, reward, next_obs_full_org, done = map(transpose_to_tensor, samples)
        
        #print("maddpg obs full org")
        #print(next_obs_full_org)
        #print(" ")


 
        obs_full = torch.stack(obs_full_org)
        # next_obs_full_org has dimension of 2 tensors of [Batch [1X24]]
        next_obs_full = torch.stack(next_obs_full_org)
        # next_obs_fullhas dimension of 1 tensors of [[Batch [1X24]], [Batch [1X24]]], this goes into target_act
        num_agent, batch, state_size =  next_obs_full.size()
        next_obs_full_reshape = next_obs_full.view(-1, state_size)
        obs_full_reshape = obs_full.view(-1, state_size)
        # has to reshape into 2XBatchx[1x24] for the concatnation with action
        
        #print("maddpg obs full org stack ")
        #print(next_obs_full)        
        #print(" ")        
 
        #print("maddpg obs full org stack reshape")
        #print(next_obs_full_reshape)        
        #print(" ")     
        
        # we are updating one agent at a time
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        # torch.cat combines all the lists into one list, i.e. states and actions from both agents combined into one list
        
        target_actions = self.target_act(next_obs_full)  
        # this returns shape of 2 tensors of batch x [1x2]
        
        # [tensor([[ 0.2826,  0.5362],
        # [ 0.2257,  0.6928],
        # [ 0.1520,  0.1303]]), tensor([[ 0.5996,  0.4146],
        # [ 0.2894,  0.1135],
        # [ 0.4787,  0.2718]])]

        target_actions = torch.cat(target_actions)
        # this returns shape of 1 tensors of 2xbatch x [1x2]
        # target_actions stacked
        # tensor([[ 0.2826,  0.5362],
        # [ 0.2257,  0.6928],
        # [ 0.1520,  0.1303],
        # [ 0.5996,  0.4146],
        # [ 0.2894,  0.1135],
        # [ 0.4787,  0.2718]])
        """ 
        target_critic_input:        
        use next obs states,
        use target_act to obtain actions from both agents
        Combine both states and actions as input to agent.target_critic
        This is used to obtain y from target critic to train critic
        """
        
        
        
        #print(" ")        
        #print("next_obs_full")         
        #print(next_obs_full)        
        #print(" ")        
        #print("next_obs_full.t()")        
        #print(next_obs_full.t())  
        #print(" ")        
        #print("target_actions")        
        #print(target_actions)    
        #print(" ")   
        #print("next_obs_full_reshape",next_obs_full_reshape.t().size() )
        #print("target_actions",target_actions.t().size() )   
        
        target_critic_input = torch.cat((next_obs_full_reshape.t(),target_actions.t())).t()
        
        #print("maddpg target crit input")
        #print(target_critic_input)    
        
        target_critic_input = target_critic_input.view(num_agent, batch, target_critic_input.size()[-1])
       
        
        
        #print("maddpg target crit input reshape")
        #print(target_critic_input)       
        
        target_critic_input = torch.cat(list(map(torch.cat,zip(*target_critic_input)))).view(batch,-1)
        
        #print("maddpg target crit input zip")
        

        #print(target_critic_input)
        #print(" ")        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        #print("q_next")
        #print(q_next)
        
        # this returns one y per batch, this is the actual 'y' for that agent
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        #print("y")
        #print(y)        
        
        """ 
        critic_input:        
        use current obs states,
        use current actions as given from sample experience
        Combine both states and actions as input to agent.critic
        This is used to obtain q from critic to compare to the y from critic_target
        """
        
        
        action = torch.cat(action)
        
        #print("obs_full_reshape.t()",obs_full_reshape.t().size() )
        #print("action.t()",action.t().size() )   
        # crit_input is using current states, target_crit is using next states
        critic_input = torch.cat((obs_full_reshape.t(), action.t())).t().to(device)
        
        critic_input = critic_input.view(num_agent, batch, critic_input.size()[-1])

        critic_input = torch.cat(list(map(torch.cat,zip(*critic_input)))).view(batch,-1)
        
        
        #print("maddpg agent crit input")
        #print(critic_input)
        #print(" ")        
        
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        
        """ 
        q_input2:        
        use current obs states,
        obtain the actions that our current actors would have gotten, not the actions as given from experience
        Combine both states and actions as input to agent.critic
        This is used to obtain q from critic to update the current agent.actor
        """        
        
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs_full_org) ]
        
        
        #print("maddpg agent q input")
        #print(q_input)      
        #print(" ")        
        q_input = torch.cat(q_input)
        #print("maddpg agent q input cat")
        #print(q_input)
        #print(" ")        
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full_reshape.t(), q_input.t())).t()        
        q_input2 = q_input2.view(num_agent, batch, q_input2.size()[-1])
        q_input2 = torch.cat(list(map(torch.cat,zip(*q_input2)))).view(batch,-1)
                
        #print("maddpg agent q2 input cat")
        #print(q_input2)     
        #print(" ")        
        # get the policy gradient
        actor_loss = - agent.critic(q_input2).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),1)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()


    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            



class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,transition):
        """push into the buffer"""
        

        self.deque.append(transition)
        
        # previously had multiple environments so need the below to pick each one from each env
        # DO NOT USE IN SINGLE ENV, because it ends up decoupling the different agents
    
        #for item in input_to_buffer:
        #    print("item")
        #    print(item)
        #    self.deque.append(item)
        # instead, append the whole transition directly without transpose to buffer

        
        
    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)



