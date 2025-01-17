
import collections
from gym.spaces import discrete
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import copy
from ActorCritic import ActorCritic

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        



class PPO:
    def __init__(self, 
        lr_actor, lr_critic, gamma, K_epochs, eps_clip, continuous_action_dim:int,discrete_action_dim:dict,action_std_init=0.6):

        # self.has_continuous_action_space = has_continuous_action_space
        self.input_target_size=640
        self.output_target_size:dict={"continuous":continuous_action_dim,"discrete":{"VM_choice":discrete_action_dim['VM_choice'],"task_index":20}}
        #这里需要变更，以统一action输出的维度
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dim=discrete_action_dim
        # self.discrete_space_splits:list=discrete_space_splits

        if continuous_action_dim:   
            self.action_std = action_std_init
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(  action_std_init,continuous_action_dim,discrete_action_dim,self.input_target_size,self.output_target_size).to(device)
        # TODO 考虑对于optimizer如何做这个参数提取的设置
        # self.optimizer = torch.optim.Adam([
        #         {'params': self.policy.shared_layers.parameters(),'lr': lr_actor},
        #         {'params': self.policy.continuous_mean_layer.parameters(),'lr': lr_actor},
        #         {'params': self.policy.discrete_logits_layer.parameters(),'lr': lr_actor},
        #         {'params': self.policy.VM_choice_output.parameters(),'lr': lr_actor},
        #         {'params': self.policy.task_index_output.parameters(),'lr': lr_actor},
        #         {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        #     ])
        self.actor_optimizer = torch.optim.Adam([
            {'params': self.policy.shared_layers.parameters(),'lr': lr_actor},
            {'params': self.policy.continuous_mean_layer.parameters(),'lr': lr_actor},
            {'params': self.policy.discrete_logits_layer.parameters(),'lr': lr_actor},
            {'params': self.policy.VM_choice_output.parameters(),'lr': lr_actor},
            {'params': self.policy.task_index_output.parameters(),'lr': lr_actor}
        ])
        self.critic_optimizer = torch.optim.Adam([
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(  action_std_init,continuous_action_dim,discrete_action_dim,self.input_target_size,self.output_target_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.continuous_action_dim:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.continuous_action_dim:
            self.action_std = self.action_std *action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def state_OrderedDict_to_list(self, ordereddict:collections.OrderedDict):   
        state_list=[]
        state_dim=0
        for key in ordereddict.keys():
            if isinstance(ordereddict[key],collections.OrderedDict):
                state_list.extend(self.state_OrderedDict_to_list(ordereddict[key])[0])
                state_dim+=self.state_OrderedDict_to_list(ordereddict[key])[1]
            else:
                state_list.extend(ordereddict[key])
                state_dim+=len(ordereddict[key])
        return state_list,state_dim

    def state_OrderedDict_to_tensor(self, state:collections.OrderedDict):
        state_tasks_list=self.state_OrderedDict_to_list(state['Node']['tasks'])[0]
        state_intervals_list=self.state_OrderedDict_to_list(state['Node']['intervals'])[0]

        state_list=state_tasks_list+state_intervals_list
        state_numpy:np.ndarray=np.nan_to_num(np.array(state_list))#去除inf
        state_tensor:torch.Tensor = torch.FloatTensor(state_numpy).to(device)
        return state_tensor
    
    def calculate_state_dim(self,state:collections.OrderedDict):
        tasks_dim=self.state_OrderedDict_to_list(state['Node']['tasks'])[1]
        intervals_dim=self.state_OrderedDict_to_list(state['Node']['intervals'])[1]
        state_dim:dict={'tasks':tasks_dim,'intervals':intervals_dim}
        return state_dim

    def select_action(self, state:collections.OrderedDict):
        # 需要根据观察结果的连续和离散部分分别处理，需要进行改造
        #对state进行处理，包括更新state_dim
        with torch.no_grad():
            state_tensor=self.state_OrderedDict_to_tensor(state)
            state_dim:dict=self.calculate_state_dim(state)
            action_done_numpy:np.ndarray=state['Node']['tasks']['VM_choice']+1
            action_mask_numpy:np.ndarray=np.array(action_done_numpy>0,dtype=int)
            action_mask_tensor:torch.Tensor=torch.tensor(action_mask_numpy,dtype=bool).to(device)  
            #如果默认值-1更改的话action_mask要更改
            action, action_logprob = self.policy_old.act(state_tensor,state_dim,action_mask_tensor)
            state_input_tensor=self.policy_old.state_padding(state_tensor,state_dim)

        self.buffer.states.append(state_input_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        #对action处理之后再进行返回
        # 试验了一下，发现传过来的Tensor分别代表的是：
        action_ret=collections.OrderedDict()
        name_list=['begin_point','VM_choice','task_index']
        for index in range(len(name_list)):
            if index<self.continuous_action_dim:
                action_ret[name_list[index]]=action[index].item()
            else:
                action_ret[name_list[index]]=np.int32(action[index].item())
        return action_ret

    def critic_forward(self):
        with torch.no_grad():
            state:torch.Tensor=self.buffer.states[-1]

            if len(self.buffer.states)!=1:#不为空的情况下
                state_before=self.buffer.states[-2]
            else:
                state_before=torch.zeros_like(state).to(device)
            input_tensor=torch.cat((state_before,state),dim=0)

            return self.policy_old.critic(input_tensor)
        
    def update(self,critic_only=False):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        zero_discounted_reward:bool = False
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if zero_discounted_reward:
                discounted_reward = 0
                zero_discounted_reward=False
            if is_terminal:
                zero_discounted_reward:bool = True
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(input=torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs: torch.Tensor = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios.T * advantages
            surr2 = torch.clamp(ratios.T, 1-self.eps_clip, 1+self.eps_clip) * advantages


            if critic_only:
                critic_loss = torch.mean(self.MseLoss(state_values, rewards))
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_old.continuous_mean_layer.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.policy_old.discrete_logits_layer.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.policy_old.VM_choice_output.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.policy_old.task_index_output.parameters(), 0.5)
                self.critic_optimizer.step()
            else:
                loss = -torch.min(surr1, surr2) + 0.2*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
                self.critic_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy_old.continuous_mean_layer.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(self.policy_old.discrete_logits_layer.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(self.policy_old.VM_choice_output.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(self.policy_old.task_index_output.parameters(), 0.1)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

            # final loss of clipped objective PPO
            # loss = -torch.min(surr1, surr2) + 0.2*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            

            # # 策略网络的损失函数
            # # actor_loss = torch.mean(-torch.min(surr1, surr2))
            # # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            # critic_loss = torch.mean(self.MseLoss(state_values, rewards))
            # # print("critic_loss:",critic_loss)

            # #原文链接：https://blog.csdn.net/dgvv4/article/details/129496576

            # # take gradient step
            # # self.optimizer.zero_grad()
            # # loss.mean().backward()
            # # self.optimizer.step()
            # # 梯度清0
            # if not critic_only:
            #     self.actor_optimizer.zero_grad()
            
            # self.critic_optimizer.zero_grad()
            # # 反向传播
            # # actor_loss.backward()
            # # critic_loss.backward()
            # if critic_only:
            #     critic_loss.backward()
            # else:
            #     loss.mean().backward()
            # # torch.nn.utils.clip_grad_norm_(self.policy_old.shared_layers.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy_old.continuous_mean_layer.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy_old.discrete_logits_layer.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy_old.VM_choice_output.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy_old.task_index_output.parameters(), 0.5)
            # # 梯度更新
            # if not critic_only:
            #     self.actor_optimizer.step()
            # self.critic_optimizer.step()

            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
