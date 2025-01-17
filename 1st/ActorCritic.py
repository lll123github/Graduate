
import collections
from gym.spaces import discrete
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import copy

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

class ActorCritic(nn.Module):
    def __init__(self, action_std_init,continuous_action_dim=0, discrete_action_dim=dict(),input_target_size=640,output_target_size=dict()):
        super(ActorCritic, self).__init__()
        self.input_target_size=input_target_size
        '''展平输入的维度，用于将输入的维度展平为一个向量，以便于进行全连接层的处理'''
        self.output_target_size:dict=output_target_size
        # self.state_dim=state_dim
        # self.tasks_dim=state_dim['tasks']
        # self.intervals_dim=state_dim['intervals']

        self.continuous_action_dim:int = continuous_action_dim
        self.discrete_action_dim:dict= discrete_action_dim
        '''指的是输出过程是否包括连续的部分，如果是连续的部分，那么输出的是一个均值和方差，如果是离散的部分，那么输出的是一个概率分布'''

        '''指的是离散的部分的分割点，用于将输出的概率分布进行分割，以便于在act和evaluate的时候进行处理'''
        
        if continuous_action_dim:
            self.continuous_action_dim = continuous_action_dim
            self.continuous_action_var = torch.full((continuous_action_dim,), action_std_init * action_std_init).to(device)


        # actor
        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_target_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        if continuous_action_dim:
            self.continuous_mean_layer = nn.Sequential(
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, continuous_action_dim),
                nn.Tanh()
            )
            
        if discrete_action_dim:

            self.discrete_logits_layer=nn.Sequential(
                nn.Linear(256, 128),
                nn.Tanh()
            )
            self.VM_choice_output=nn.Sequential(
                nn.Linear(128, output_target_size['discrete']['VM_choice']),
                nn.Softmax(dim=-1)
            )
            self.task_index_output=nn.Sequential(
                nn.Linear(128, out_features=output_target_size['discrete']['task_index']),
                nn.Softmax(dim=-1)
            )


        # critic
        self.critic = nn.Sequential(
                nn.Linear(self.input_target_size*2, 512),#输入两个状态
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256,128),
                nn.Tanh(),
                nn.Linear(128, out_features=1)
            )
        
    def set_action_std(self, new_action_std):
        if self.continuous_action_dim:
            self.continuous_action_var = torch.full((self.continuous_action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def state_padding(self,state_input:torch.Tensor,state_dim:dict):
        state=copy.deepcopy(state_input)
        
        tasks_dim:int=state_dim['tasks']
        intervals_dim:int=state_dim['intervals']
        #进行一下全零填充
        pad_center_size=self.input_target_size-tasks_dim-intervals_dim
        if pad_center_size<0:
            print("tasks_dim",tasks_dim)
            print("intervals_dim",intervals_dim)
            raise ValueError('The target size is too small to fit the input size')
        #如果输入的tensor维度是1的话
        if len(state.shape)==1:
            pad_tensor:torch.Tensor=torch.full((pad_center_size,),-1).to(device)
            left_tensor,right_tensor=torch.split(state,[tasks_dim,intervals_dim],dim=0)
            left_tensor=left_tensor.to(device)
            right_tensor=right_tensor.to(device)
            input_tensor=torch.cat((left_tensor,pad_tensor,right_tensor),dim=0)
        else:
            pad_tensor:torch.Tensor=torch.full((state.shape[0],pad_center_size),-1).to(device)
            left_tensor,right_tensor=torch.split(state,[tasks_dim,intervals_dim],dim=1)
            left_tensor=left_tensor.to(device)
            right_tensor=right_tensor.to(device)
            input_tensor=torch.cat((left_tensor,pad_tensor,right_tensor),dim=1)
        
        return input_tensor


    
    def forward(self,state_input:torch.Tensor,state_dim:dict):
        if state_dim is None:
            state_dim={'tasks':0,'intervals':self.input_target_size}
        
        input_tensor:torch.Tensor=self.state_padding(state_input,state_dim=state_dim)
        input_tensor=input_tensor.to(device)
        shared_output:torch.Tensor = self.shared_layers(input_tensor).to(device)
        if self.continuous_action_dim:
            self.continuous_mean:torch.Tensor = self.continuous_mean_layer(shared_output)
        else:
            self.continuous_mean=torch.Tensor([])
        if self.discrete_action_dim:
            self.discrete_middle:torch.Tensor = self.discrete_logits_layer(shared_output)
            self.vm_logits:torch.Tensor=self.VM_choice_output(self.discrete_middle)
            self.task_logits:torch.Tensor=self.task_index_output(self.discrete_middle)
            #得到离散两个输出，之后进行拼接。考虑到批量和单独的情况，使用不同的dim
            if len(self.task_logits.shape)==1:
                self.discrete_logits=torch.cat((self.vm_logits,self.task_logits),dim=0)
            else:
                self.discrete_logits=torch.cat((self.vm_logits,self.task_logits),dim=1)
        else:
            self.discrete_logits=torch.Tensor([])
        return self.continuous_mean, self.discrete_logits
    
    def act(self, state:torch.Tensor,state_dim:dict,action_mask:torch.Tensor):
        #TODO 需要根据结观察果的数据类型的离散部分和连续部分进行改造
        forward_result=self.forward(state,state_dim)
        
        if self.continuous_action_dim:
            continuous_action_mean = forward_result[0]
            cov_mat = torch.diag(self.continuous_action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(continuous_action_mean, cov_mat)
            continuous_action:torch.Tensor = dist.sample()#应该要是一个一维的Tensor
            continuous_action_logprob:torch.Tensor = dist.log_prob(continuous_action)#对于连续的分布，返回的是该点处概率密度的对数值
            #这边有针对应用场景的处理，如果是其它的有多维的话可能不能这样使用
            continuous_action=torch.reshape(continuous_action, (1,))
            continuous_action_logprob=torch.reshape(continuous_action_logprob, (1,))
        if self.discrete_action_dim:
            discrete_action_probs = forward_result[1]
            discrete_action_probs_splits:tuple=torch.split(discrete_action_probs, list(self.output_target_size['discrete'].values()), dim=0)
            #分割离散的部分
            discrete_action_list = []
            discrete_action_logprob_list = []

            for splits_index,used_size in enumerate(list(self.discrete_action_dim.values())):
                # if splits_index==1:#对于task_index的处理，遮盖已经处理过的task
                    # discrete_action_probs_splits[splits_index]=discrete_action_probs_splits[splits_index]*action_mask
                split:torch.Tensor=discrete_action_probs_splits[splits_index].to(device)
                split[used_size:]=1e-4
                dist = Categorical(split.cuda())
                action = dist.sample()
                logprob = dist.log_prob(value=action)
                discrete_action_list.append(action)
                discrete_action_logprob_list.append(logprob)

            discrete_action = torch.Tensor(discrete_action_list).to(device)
            discrete_action_logprob = torch.Tensor(discrete_action_logprob_list).to(device)



        # action = dist.sample()
        # action_logprob = dist.log_prob(action)
        # TODO 之后需要考虑一下离散或者连续的变量为空的时候的情况
        # 拼接#TODO 检查一下dim是不是正确的，需要根据批次来确定
        action = torch.cat((continuous_action, discrete_action), dim=0)
        action_logprob = torch.cat((continuous_action_logprob, discrete_action_logprob), dim=0)#对于离散的分布，返回的是该点处概率的对数值
        
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action:torch.Tensor):
        # 需要根据结观察果的数据类型的离散部分和连续部分
        if len(action.shape)==1:
            action=torch.unsqueeze(action,0)
            
        continuous_action,discrete_action=torch.split(action, [self.continuous_action_dim, len(self.output_target_size['discrete'].values())], dim=1)
        
        continuous_action_mean,discrete_action_probs=self.forward(state,None)
        if len(continuous_action_mean.shape)==1:
            continuous_action_mean=torch.unsqueeze(continuous_action_mean,0)
        if len(discrete_action_probs.shape)==1:
            discrete_action_probs=torch.unsqueeze(discrete_action_probs,0)


        if self.continuous_action_dim:
            
            continuous_action_var = self.continuous_action_var.expand_as(continuous_action_mean)
            cov_mat = torch.diag_embed(continuous_action_var).to(device)#获得协方差矩阵。默认多个随机的变量之间独立
            dist = MultivariateNormal(continuous_action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.continuous_action_dim == 1:
                continuous_action = continuous_action.reshape(-1, self.continuous_action_dim)
            
            continuous_action_logprobs:torch.Tensor = dist.log_prob(continuous_action)
            if self.continuous_action_dim == 1:
                continuous_action_logprobs = torch.unsqueeze(continuous_action_logprobs, dim=0)
            continuous_dist_entropy:torch.Tensor = dist.entropy()

        if self.discrete_action_dim:
            
            discrete_action_probs_splits:tuple=torch.split(discrete_action_probs, list(self.output_target_size['discrete'].values()), dim=1)
            discrete_action_logprobs_list = []
            discrete_dist_entropy:torch.Tensor = torch.zeros(discrete_action_probs.shape[0]).to(device)

            for splits_index,used_size in enumerate(list(self.discrete_action_dim.values())):
                split:torch.Tensor=discrete_action_probs_splits[splits_index].clone()
                if min(split[:,used_size:].size())!=0:#判断tensor是否为空
                    # split[:,used_size:]=1e-4
                    #注意这个报错
                    # RuntimeError: Output 0 of SliceBackward0 is a view and is being modified inplace. This view is the output of a function that returns multiple views. Such functions do not allow the output views to be modified inplace. You should replace the inplace operation by an out-of-place one.
                    split[:,used_size:]=1e-4#对于不会出现的情况赋以一个很小的概率
                dist = Categorical(split.cuda())
                logprob = dist.log_prob(discrete_action[:,splits_index])
                discrete_action_logprobs_list.append(logprob)
                discrete_dist_entropy+=dist.entropy()

            discrete_action_logprobs = torch.stack(discrete_action_logprobs_list)
            
            
            # dist = Categorical(discrete_action_probs)
            # #TODO If probs is N-dimensional, the first N-1 dimensions are treated as a batch of relative probability vectors.
            # discrete_action_logprobs = dist.log_prob(discrete_action)
            # discrete_dist_entropy = dist.entropy()

        if len(state.shape)==1:
            state=torch.unsqueeze(state,0)

        state_shift=torch.cat((torch.zeros(1,self.input_target_size).to(device),state),dim=0)
        state_input=torch.cat((state,state_shift[:state.shape[0]]),dim=1)
        state_values = self.critic(state_input)
        
        action_logprobs=torch.cat((continuous_action_logprobs, discrete_action_logprobs),dim=0).T
        dist_entropy=sum(continuous_dist_entropy+discrete_dist_entropy)
        #TODO 需要考虑变量的关系考虑是否需要对协方差矩阵进行修改，以及如果将离散变量和连续变量的关系考虑进去，那么熵不是简单的相加？
        return action_logprobs, state_values, dist_entropy