# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Actor, Critic
from utilities import hard_update, gumbel_softmax, onehot_from_logits, transpose_to_tensor, transpose_list 
from torch.optim import Adam
import torch
import numpy as np
import random
import copy
# add OU noise for exploration

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'


class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic_state, in_critic_action, hidden_in_critic, hidden_out_critic, lr_actor, lr_critic, seed):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(in_actor, out_actor, seed, hidden_in_actor, hidden_out_actor).to(DEVICE)
        self.target_actor = Actor(in_actor, out_actor, seed, hidden_in_actor, hidden_out_actor).to(DEVICE)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)        
        
        self.critic = Critic(in_critic_state, in_critic_action, seed, hidden_in_critic, hidden_out_critic).to(DEVICE)        
        self.target_critic = Critic(in_critic_state, in_critic_action, seed, hidden_in_critic, hidden_out_critic).to(DEVICE)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        
        self.noise = OUNoise(out_actor, seed)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        self.updated_times = 0
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
    def act(self, state, noise=0.0):
        state = state.to(DEVICE)     
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        
        if noise > 0.0:
            action = torch.clamp(action + noise*self.noise.noise(), min=-1, max=1) 

        return action

    def target_act(self, state, noise=0.0):
        state = state.to(DEVICE)
        self.target_actor.eval()
        with torch.no_grad():
            action = self.target_actor(state)
        
        if noise > 0.0:
            action = torch.clamp(action + noise*self.noise.noise(), min=-1, max=1)        
        return action
    
    def reset(self):
        self.noise.reset()
 

   
# ----------------------------------------- updating --------------------------------------- #            
    def learn(self, samples, pred_actions, target_actions, agent_num, discount_factor=0.99):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

    
        
        states, actions, rewards, next_states, dones, full_state, next_full_state = samples
 
        # full_state has dim Batchx[1X48]
        # ---------------------------- transformations ---------------------------- #

        states = torch.stack(states)
        # states has dimension of 2 tensors of [Batch [1X24]]
        next_states = torch.stack(next_states)
        # next_states dimension of 1 tensors of [[Batch [1X24]], [Batch [1X24]]], this goes into target_act
        
        num_agent, batch, state_size =  states.size()
        #print("samples")
        #print(samples)
        #print("")
        #print("actions org")
        #print(actions)
        #print("")        
        #print("pred_actions org")
        #print(pred_actions)
        #print("")  
        #print("target_actions org")
        #print(target_actions)
        #print("")       
        #actions = torch.cat(actions, dim = 0)     
        actions = torch.cat(actions, dim = -1)
        

        #print("actions transpose")
        #print(actions)
        #print("")  
        #print("next_full_state")
        #print(next_full_state)
        
        #print("dones")
        #print(dones)
        

        
        target_actions = torch.cat(target_actions, dim = -1) 
        #print(next_full_state.shape)
        #print(target_actions.shape)        
        #print(torch.cat((next_full_state, target_actions), dim= -1))
        #print(torch.cat((full_state, actions), dim= -1))

        q_next = self.target_critic(next_full_state.to(DEVICE), target_actions.to(DEVICE))
        #print("q_next", q_next)
        #print("discount_factor * q_next * (1 - dones[agent_num].view(-1, 1))",discount_factor * q_next * (1 - dones[agent_num].view(-1, 1)))
        #print("rewards[agent_num].view(-1, 1) ",rewards[agent_num].view(-1, 1) )
        

        Q_crit_targets = rewards[agent_num].view(-1, 1) + discount_factor * q_next * (1 - dones[agent_num].view(-1, 1))
        #print("Q_crit_targets",Q_crit_targets)       
        # ---------------------------- update critic ---------------------------- #
        # each current critic takes all the states and all the actions

        
        Q_cur_crit = self.critic(full_state.to(DEVICE), actions.to(DEVICE))
        #print("Q_cur_crit")
        #print(Q_cur_crit)    

        critic_loss = torch.nn.functional.mse_loss(Q_cur_crit, Q_crit_targets.detach())
        #print()
        #print("start learning",agent_num)

        #print("critic_loss", critic_loss)
        # Minimize the loss
        #print("critic.parameters")
        #print(list(self.critic.parameters())[0])
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.critic.parameters(), 1)        
        self.critic_optimizer.step()
        #print("critic.parameters after update")
        #print(list(self.critic.parameters())[0])
        #---------------------------- get pred actions NOT UPDATED----------------------------#
        
        
        cur_agent_pred_actions = self.actor(states[agent_num]) 
        pred_actions_cur = [cur_agent_pred_actions  if i == agent_num \
                            else pred_actions[i].detach()
                            for i in range(num_agent)]
        pred_actions_cur = torch.cat(pred_actions_cur, dim = -1) 
        #print("cur_agent_pred_actions")
        #print(cur_agent_pred_actions)
        #print("pred_actions_cur")
        #print(pred_actions_cur)
        #print("agent_num")
        #print(agent_num)
        #print("pred_actions provided")
        #print(pred_actions)
                
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss

        actor_loss = - self.critic(full_state.to(DEVICE), pred_actions_cur.to(DEVICE)).mean()
                
        # Minimize the online critic loss
        #print(" actor.parameters")
        #print(list(self.actor.parameters())[0])        
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.actor.parameters(), 1)            
        self.actor_optimizer.step()
        #print("actor_loss", actor_loss)

        #print(" actor.parameters after update")
        #print(list(self.actor.parameters())[0]  ) 
        #print("fininshed learning")
        #print()
        self.updated_times += 1
        if self.updated_times % 2000 == 0:
            print("actor", actor_loss.item(), "critic",critic_loss.item())
            
        
        

# ---------------------------- below no need update ---------------------------- #         
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def noise(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # normal noises        
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        # random noises
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return torch.tensor(self.state).float()
