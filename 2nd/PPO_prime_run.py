import os
import glob
import time
import collections
from datetime import datetime
from CustomEnv import CustomEnv
from gym import spaces

import torch
import numpy as np

import gymnasium as gym
# import roboschool

from PPO import PPO
import copy

import argparse
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--env', type=str, default='')
#     return parser.parse_args()

################################### Training ###################################
# def train(args=get_args()):
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    # env_name = "RoboschoolWalker2d-v1"
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda') 
        torch.cuda.empty_cache()

    # env_name = args.env
    env_name="TaskIntervalEnv"
    
    # has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 30                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timesteps > max_training_timesteps
    only_critic_training=int(10000)
    print_freq = max_ep_len * 50        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 50           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.01                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.8        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.001                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(1e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    # update_timestep = max_ep_len * 1      # update policy every n timesteps
    update_timestep=25
    K_epochs = 10               # update policy for K epochs in one PPO update 计算一次损失

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0000005       # learning rate for actor network
    lr_critic = 0.000025       # learning rate for critic network
    lr_actor_decay_rate = 0.8
    lr_critic_decay_rate = 0.8
    lr_actor_decay_freq = int(1e6)

    random_seed = 0        # set random seed if required (0 = no random seed)
    # random_seed = args.seed         # set random seed if required (0 = no random seed)
    #####################################################

    # print("training environment name : " + env_name)

    # env = gym.make(env_name)
    env=CustomEnv(5,20)

    # state space dimension
    tasks_dim=gym.spaces.flatdim(env.observation_space['Node']['tasks'])
    intervals_dim=gym.spaces.flatdim(env.observation_space['Node']['intervals'])


    state_dim={'tasks':tasks_dim,'intervals':intervals_dim}

    # action space dimension
    # if has_continuous_action_space:
    discrete_action_dim:dict=dict()
    continuous_action_dim = 0
    # 对ppo_agent进行一个修正，使得可以对接上环境的输出
    for space_name, sub_space in env.action_space.spaces.items():
        if isinstance(sub_space, gym.spaces.Discrete):
            #注意这边一定要以gymnasium开头，不然会判断为False
            discrete_action_dim[space_name]=gym.spaces.flatdim(sub_space)
        elif isinstance(sub_space, gym.spaces.Box):
            continuous_action_dim += gym.spaces.flatdim(sub_space)
    #需要根据空间的定义来进行改变，确认是个数还是维度
    # else:
    #     # action_dim = env.action_space.n
    #     # 正负向动作
    #     action_dim = env.target_section_length * 2

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    # log_dir = ""
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    log_dir = './' + env_name + '/'
    log_dir = log_dir + f'prime_seed_{random_seed}_{datetime.now().strftime("%m%d_%H-%M-%S")}/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log" + ".csv"

    print("current logging run number for " + env_name + " : ")
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    # directory = ""
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    directory = './' + env_name + '/'
    directory = directory + f'prime_seed_{random_seed}_{datetime.now().strftime("%m%d_%H-%M-%S")}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, random_seed)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("init state space dimension : ", state_dim)
    
    print("continuous action space dimension : ", continuous_action_dim)
    print("discrete action space dimension : ", discrete_action_dim)
    print("--------------------------------------------------------------------------------------------")
    if continuous_action_dim:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    if discrete_action_dim:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO( lr_actor, lr_critic, gamma, K_epochs, eps_clip, continuous_action_dim,discrete_action_dim, action_std)

    
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward,steps\n')
    

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    print_running_step = 0
    print_done_num=0
    print_episode_num=0

    log_running_reward = 0
    log_running_episodes = 0
    log_running_step = 0

    time_step = 0
    i_episode = 0

    
    ppo_agent.load('./TaskIntervalEnv/prime_seed_0_0501_16-03-22/PPO_TaskIntervalEnv_0_3000000_0501_23-22-52.pth')
    # training loop
    while time_step <= max_training_timesteps:
        interval_node_num=int(np.random.randint(8,50))
        task_node_num=int(np.random.randint(1,20))
        env.reset(task_node_num,interval_node_num)
        # env.reset(int(np.random.randint(1,20)),5)
        # 不建议靠5太近，因为可能会导致难随机出分割点
        state=copy.deepcopy(env.state)
        state.pop('Link')#这里会修改env里面的结构，不合适
        #TODO 对state进行一个修正，使得环境可以成功对接上PPO的训练过程TODO

        # 对ppo_agent进行一个修正，使得可以对接上环境的输出
        ppo_agent.continuous_action_dim=0

        for space_name, sub_space in env.action_space.spaces.items():
            if isinstance(sub_space, gym.spaces.Discrete):
                #注意这边一定要以gymnasium开头，不然会判断为False
                ppo_agent.discrete_action_dim[space_name]=gym.spaces.flatdim(sub_space)
            elif isinstance(sub_space, gym.spaces.Box):
                ppo_agent.continuous_action_dim += gym.spaces.flatdim(sub_space)
            #需要根据空间的定义来进行改变，确认是个数还是维度


        current_ep_reward = 0
        current_ep_state_value=0
        output_str=""
        max_ep_len=int(task_node_num*2.5)
        for t in range(1, max_ep_len+1):
            #更新state
            state=copy.deepcopy(env.state)
            # select action with policy
            action:collections.OrderedDict = ppo_agent.select_action(state)

            # act = action // env.target_section_length
            # target = action % env.target_section_length
            # true_action = [[act, target]]

            state_ret, reward, done, output_str = env.step(action)
            # print(f"t={t}：{output_str}")

            state_value=ppo_agent.critic_forward().item()
            #不应该写作 state=state.pop('Link')，这样会导致state为Link的值
            reward = np.float32(reward)
            done: bool = done
            #更新state_dim
            # state_dim={'tasks':gym.spaces.flatdim(env.observation_space['Node']['tasks']),'intervals':gym.spaces.flatdim(env.observation_space['Node']['intervals'])}
            # print("state_dim",state_dim)

            # saving reward and is_terminals 
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            
            time_step +=1
            current_ep_reward += reward
            log_running_step += 1
            current_ep_state_value+=state_value

            # if update_timestep<only_critic_training:
            #     critic_only=True
            # else:
            critic_only=False
            
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update(critic_only)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                # log average step till last episode
                log_avg_step = log_running_step / log_running_episodes

                log_f.write('{},{},{},{}\n'.format(i_episode, time_step, log_avg_reward, log_avg_step))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0
                log_running_step = 0

            # if continuous action space; then decay action std of ouput action distribution
            if continuous_action_dim and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                # print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Average Step : {} \t\t Done Num : {}/{}".format(i_episode, time_step, print_avg_reward, print_running_step, print_done_num, print_episode_num))

                print_running_reward = 0
                print_running_episodes = 0
                print_running_step = 0
                print_done_num=0
                print_episode_num=0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path=directory + f'PPO_{env_name}_{random_seed}_{time_step}_{datetime.now().strftime("%m%d_%H-%M-%S")}.pth'
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if time_step % lr_actor_decay_freq==0:
                for index in range(len(ppo_agent.actor_optimizer.param_groups)):
                    ppo_agent.actor_optimizer.param_groups[index]['lr']*=lr_actor_decay_rate
                for index in range(len(ppo_agent.critic_optimizer.param_groups)):
                    ppo_agent.critic_optimizer.param_groups[index]['lr']*=lr_critic_decay_rate
                
            if done:
                break
        
        # if t>1 and current_ep_reward>0.0:
        # if done:
        # print(f"{done}. Episode {i_episode} finished after {t} timesteps.\t Tasks num: {task_node_num} \tOutput: {output_str}. \tReward: {current_ep_reward} \tState_value: {current_ep_state_value}")
        if done:
            print_done_num+=1
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        print_running_step += 1
        print_episode_num+=1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        
    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()