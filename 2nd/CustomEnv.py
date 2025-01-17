
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections
import copy

class CustomEnv(gym.Env):
    def __init__(self,task_node_num=8,interval_node_num=20) -> None:
        super(CustomEnv,self).__init__()
        self.lam=np.random.uniform(low=20,high=50)
        '''单位时间内到达的任务数，来生成任务到达的时间间隔'''

        self.mu=np.random.uniform(low=20,high=50)
        '''单位时间内完成的任务数，用在指数分布中来生成任务的持续时间'''

        self.phi=np.random.uniform(low=20,high=50)
        '''单位时间内接收的任务数'''

        self.normal_mean_rate=np.random.uniform(low=1.5,high=2.5)
        '''正态分布的均值'''

        self.normal_var_rate=np.random.uniform(low=0.05,high=0.15)
        '''正态分布的方差'''

        self.task_node_num=task_node_num
        self.interval_node_num=interval_node_num
        self.init_interval_node_num=interval_node_num
        self.vm_num=5
        self.observation_space:spaces.Dict=None
        self.action_space:spaces.MultiDiscrete=None
        self.initial_state:collections.OrderedDict=None
        self.state:collections.OrderedDict=None
        self.action:collections.OrderedDict=None
        # self.steps_beyond_done=None
        self.error_reward:np.float32=-10.0
        self.no_node_error_reward=-20.0
        self.no_vm_error_reward=-20.0
        self.dealed_reward=-30.0
        self.time_error_reward=-10.0
        self.usage_error_reward=-5.0
        self.done_tasks_reward=5.0
        self.done_reward=100.0
        self.inf_replace:np.float32=10000.0
        self.error_break:bool=False

        self.init_history()
                # 随机参数
        
        
        # 生成观察
        self.observationSpaceGenerator()
        self.actionSpaceGenerator()
        self.initial_state=self.observation_space.sample()
        self.stateInitModifier()
        self.state=copy.deepcopy(self.initial_state)
        self.action=None

        #重置历史
        self.init_history()

        self.tasks_modified_history:set=set()
        
        
        print("self.initial_state")
        print(self.initial_state)


    def init_history(self):
        # 记录历史
        self.history_action:list=[]
        self.history_state:list=[]
        self.history_reward:list=[]
        

    def observationSpaceGenerator(self):
        # 如果调不出来的话看看经点的算法。给计算资源设置多个阈值？
        # 生成观察
        # 还可以进行任务的优先级和先后顺序建模
        # 没有使用Graph类型是因为Graph不支持分别定义节点离散属性和连续属性？但是应该可以使用字典类型来进行弥补？（不可以，在库中被定义了）
        # self.task_node_num=np.random.randint(low=5,high=10)
        # self.interval_node_num=np.random.randint(low=5,high=10)
        self.observation_space=spaces.Dict(
            {
                "Node":spaces.Dict(
                    {
                        # 计算资源类型的节点，0：空余占用率，1：开始时刻，2：持续时长，3：结束时刻。后面三者都需要在采样的时候进行修改
                        "intervals":spaces.Dict(
                            {
                                "index":spaces.Box(
                                    low=0,high=self.interval_node_num,shape=(self.interval_node_num,),dtype=np.int32
                                ),
                                "usage":spaces.Box(
                                    low=0.0,high=1.0,shape=(self.interval_node_num,),dtype=np.float32
                                ),
                                "start":spaces.Box(
                                    low=0.0,high=self.inf_replace,shape=(self.interval_node_num,),dtype=np.float32
                                ),
                                "end":spaces.Box(
                                    low=0.0,high=self.inf_replace,shape=(self.interval_node_num,),dtype=np.float32
                                ),
                                "VM":spaces.Box(low=0,high=self.vm_num,shape=(self.interval_node_num,),dtype=np.int32)
                                
                            }
                        ),
                        # 任务类型的节点，0：占用率，1：到达时刻，2：需要的时长，3：截止时刻。后面三者都需要在采样的时候进行修改
                        "tasks":spaces.Dict(
                            {
                                "index":spaces.Box(
                                    low=0,high=self.task_node_num,shape=(self.task_node_num,),dtype=np.int32
                                ),
                                "usage":spaces.Box(
                                    low=0.0,high=1.0,shape=(self.task_node_num,),dtype=np.float32
                                ),
                                "arrival":spaces.Box(
                                    low=0.0,high=self.inf_replace,shape=(self.task_node_num,),dtype=np.float32
                                ),
                                "duration":spaces.Box(
                                    low=0.0,high=self.inf_replace,shape=(self.task_node_num,),dtype=np.float32
                                ),
                                "deadline":spaces.Box(
                                    low=0.0,high=self.inf_replace,shape=(self.task_node_num,),dtype=np.float32
                                ),
                                "VM_choice":spaces.Box(
                                    low=0,high=self.vm_num,shape=(self.task_node_num,),dtype=np.int32
                                )
                            }
                        )#或许之后可以使用Sequence类型？
                    }
                ),
                # "Edge":spaces.Discrete(1),
                "Link":spaces.Sequence(spaces.MultiDiscrete(np.array([self.task_node_num,self.interval_node_num],dtype=np.int32)))
            }
        )

    def stateInitModifier(self):
        # 制作间隙
        # 对这么多个间隙进行一下分割，分成vm_num组间隙
        # print("在stateInitModifier中的interval_node_num是：{self.interval_node_num}")
        # print(self.interval_node_num)
        #需要生成不重复的随机整数
        split_set=set()
        while len(split_set)<self.vm_num-1:
            split_set.add(np.random.randint(low=1,high=self.interval_node_num-1))
            # print("split_set",split_set)

        split_set=np.array(list(split_set))
        split_set=np.array([0]+list(split_set)+[self.interval_node_num])
        split_set=np.sort(split_set)
        # print("split_set",split_set)

        self.initial_state['Node']['intervals']['index']=np.arange(self.interval_node_num)
        for vm_index in range(self.vm_num):
            task_process_time_span=np.random.exponential(scale=1.0/(self.lam+self.mu),size=(split_set[vm_index+1]-split_set[vm_index],1))
            end=np.cumsum(task_process_time_span)#累积一下，把间隙的持续时间转成时长
            start=np.delete(np.insert(end,0,0),-1)#得到每一个间隙的开始时刻
            slice_head=split_set[vm_index]
            slice_tail=split_set[vm_index+1]
            # print(start.shape,self.initial_state['Node']['intervals']['start'][slice_head:slice_tail].shape)
            
            self.initial_state['Node']['intervals']['start'][slice_head:slice_tail]=start
            self.initial_state['Node']['intervals']['end'][slice_head:slice_tail]=end
            self.initial_state['Node']['intervals']['VM'][slice_head:slice_tail]=np.array([vm_index]*(split_set[vm_index+1]-split_set[vm_index]))
            self.initial_state['Node']['intervals']['usage'][slice_tail-1]=1.0#设置一个服务器最后的空余为1.0，表示完全没有被占用
            self.initial_state['Node']['intervals']['end'][slice_tail-1]=self.inf_replace#设置一个服务器最后的结束时刻为无穷大，表示这个服务器间隙到永久


            
        # 制作任务
        # 生成任务的到达时刻
        
        task_size=(self.task_node_num,)
        arrival=np.random.exponential(scale=1.0/self.lam,size=task_size)
        arrival=np.cumsum(arrival)
        # 生成任务的持续时间
        task_process_time_span=np.random.exponential(scale=1.0/self.mu,size=task_size)
        # 任务的等待时间，保持和之前的代码一致
        task_receive_time_span:np.ndarray= np.random.exponential(scale=1/self.phi,size=task_size)#接收时长
    
        task_process_time_span=task_receive_time_span+task_process_time_span
        task_normal_mean=task_process_time_span*self.normal_mean_rate
        task_normal_var=task_process_time_span*self.normal_var_rate
        task_max_wait_time_span=np.empty(shape=task_size,dtype=np.float32)
        for index,min_wait_time in enumerate(task_process_time_span):
            while True:
                random_wait_time:np.ndarray=np.random.normal(loc=task_normal_mean[index],scale=task_normal_var[index],size=1)
                random_wait_time=float(random_wait_time[0])#因为只有一个元素，所以可以这样操作
                if random_wait_time>min_wait_time:
                    task_max_wait_time_span[index]=random_wait_time
                    break
        # 生成任务的截止时刻
        deadline=arrival+task_max_wait_time_span

        self.initial_state['Node']['tasks']['index']=np.arange(self.task_node_num)
        self.initial_state['Node']['tasks']['arrival']=arrival
        self.initial_state['Node']['tasks']['duration']=task_process_time_span
        self.initial_state['Node']['tasks']['deadline']=deadline
        self.initial_state['Node']['tasks']['VM_choice']=np.full(task_size,-1) 
        
        # 制作链接
        self.initial_state['Link']=()

        return


    def actionSpaceGenerator(self):
        # 动作空间
        # 选择节点进行连接，在连接的同时对节点进行分裂
        # 但是不应该使用discrete类型，因为连接节点需要选择节点，分离节点也需要选择节点
        # self.action_space=spaces.Sequence(spaces.MultiDiscrete(np.array([self.task_node_num,self.interval_node_num],dtype=np.int32)))
        # 相比于使用动作空间，传递间隙的特征更加容易处理一些
        #动作空间是不需要随着环境进行更新的
        self.action_space=spaces.Dict(
            {
                "task_index":spaces.Discrete(self.task_node_num),
                "VM_choice":spaces.Discrete(self.vm_num),
                "begin_point":spaces.Box(low=0,high=self.inf_replace,dtype=np.float32)
            }
        )



    
    def reset(self,task_node_num,interval_node_num):

        # 生成观察
        self.task_node_num=task_node_num
        self.interval_node_num=interval_node_num
        self.observationSpaceGenerator()
        self.actionSpaceGenerator()
        self.initial_state=self.observation_space.sample()
        self.stateInitModifier()
        self.state=copy.deepcopy(self.initial_state)
        self.action=None

        #重置历史
        self.init_history()

        self.tasks_modified_history:set=set()

        # 重置环境状态
        # self.state=self.initial_state
        
        # self.steps_beyond_done=None

        # 返回初始观察
        return self.state


    
    def step(self,action:collections.OrderedDict):
        
        # 执行动作
        self.action=action
        self.history_action.append(action)

        # action 的格式和动作空间的格式一致。
        # 先检查节点是否存在
        action_task_index:np.int32=np.int32(action["task_index"])
        if (not action_task_index in self.state['Node']['tasks']['index']):
            self.history_state.append(self.state)
            self.history_reward.append(self.error_reward)
            return self.state,self.no_node_error_reward,self.error_break,"节点不存在"
        
        #检查VM是否存在
        if (not action["VM_choice"] in self.state['Node']['intervals']['VM']):
            self.history_state.append(self.state)
            self.history_reward.append(self.error_reward)
            return self.state,self.no_vm_error_reward,self.error_reward,"VM不存在"
        # 检查是否已经处理过相同的节点
        # TODO 之后可以改进，可以取消tasks的VM_choice属性，改为在link中记录
        if action_task_index in self.tasks_modified_history:
            self.history_state.append(self.state)
            self.history_reward.append(self.error_reward)
            return self.state,self.dealed_reward,self.error_break,"节点已经处理过"
        
        
        
        # 检查传递进来的间隙是否符合要求
        begin_point:np.float32=action["begin_point"]
        end_point:np.float32=action["begin_point"]+self.state['Node']['tasks']['duration'][action_task_index]
        arrival_point:np.float32=self.state['Node']['tasks']['arrival'][action_task_index]

        if (begin_point>=end_point) or (begin_point<arrival_point):
            # print("begin_point",begin_point)
            # print("end_point",end_point)
            # print("arrival_point",arrival_point)
            self.history_state.append(self.state)
            self.history_reward.append(self.error_reward)
            return self.state,self.time_error_reward,self.error_break,"时间错误，越俎代庖"


        # 找到对应占用的间隙
        intervals:collections.OrderedDict=self.state['Node']['intervals']
        vm_mask:np.ndarray=(intervals['VM']==action['VM_choice'])
        vm_intervals_index:np.ndarray=intervals['index'][vm_mask]
        vm_intervals_start:np.ndarray=intervals['start'][vm_mask]
        vm_intervals_end:np.ndarray=intervals['end'][vm_mask]
        vm_intervals_usage:np.ndarray=intervals['usage'][vm_mask]

        time_mask:np.ndarray=(vm_intervals_start<=end_point)*(vm_intervals_end>=begin_point)
        intervals_index:np.ndarray=vm_intervals_index[time_mask]
        intervals_start:np.ndarray=vm_intervals_start[time_mask]
        intervals_end:np.ndarray=vm_intervals_end[time_mask]
        intervals_usage:np.ndarray=vm_intervals_usage[time_mask]

        

        if intervals_index.size==0:
            print("vm_interval_index",vm_intervals_index)
            print("vm_interval_start",vm_intervals_start)
            print("vm_interval_end",vm_intervals_end)
            print("vm_interval_usage",vm_intervals_usage)
            raise ValueError("间隙不存在，请检查间隙定义和修改情况")
            return self.state,self.error_reward,self.error_break,"间隙不存在"
        
        # 检查占用率
        task_usage=self.state['Node']['tasks']['usage'][action_task_index]
        usage_error:np.ndarray=(intervals_usage<task_usage)
        if usage_error.any():
            return self.state,self.usage_error_reward,self.error_break,"占用率错误"
        #首末节点拆分
        # print("interval_index",intervals_index)
        # print("interval_start",intervals_start)
        # print("interval_end",intervals_end)
        # print("interval_usage",intervals_usage)
        # print("VM_choice",action['VM_choice'])
        # print(self.state['Node']['intervals']['VM'][intervals_index])
        # 找到首节点
        first_interval_sqindex:int=np.argmin(intervals_start)#注意是筛选之后列表的编号
        # 找到末节点
        last_interval_sqindex:int=np.argmax(intervals_end)

        # 拆分首节点
        if intervals_start[first_interval_sqindex]<begin_point:
            # 生成新的节点
            self.state['Node']['intervals']['index']=np.append(self.state['Node']['intervals']['index'],self.interval_node_num)
            self.state['Node']['intervals']['start']=np.append(self.state['Node']['intervals']['start'],values=intervals_start[first_interval_sqindex])
            self.state['Node']['intervals']['end']=np.append(self.state['Node']['intervals']['end'],values=begin_point)
            self.state['Node']['intervals']['usage']=np.append(self.state['Node']['intervals']['usage'],values=intervals_usage[first_interval_sqindex])
            self.state['Node']['intervals']['VM']=np.append(self.state['Node']['intervals']['VM'],values=action['VM_choice'])
            self.interval_node_num+=1
            # 修改原节点
            self.state['Node']['intervals']['start'][intervals_index[first_interval_sqindex]]=begin_point
            # 占用率统一扣除

        # 拆分末节点
        if intervals_end[last_interval_sqindex]>end_point:
            # 生成新的节点
            self.state['Node']['intervals']['index']=np.append(self.state['Node']['intervals']['index'],self.interval_node_num)
            self.state['Node']['intervals']['start']=np.append(self.state['Node']['intervals']['start'],values=end_point)
            self.state['Node']['intervals']['end']=np.append(self.state['Node']['intervals']['end'],values=intervals_end[last_interval_sqindex])
            self.state['Node']['intervals']['usage']=np.append(self.state['Node']['intervals']['usage'],values=intervals_usage[last_interval_sqindex])
            self.state['Node']['intervals']['VM']=np.append(self.state['Node']['intervals']['VM'],values=action['VM_choice'])
            self.interval_node_num+=1
            # 修改原节点
            self.state['Node']['intervals']['end'][intervals_index[last_interval_sqindex]]=end_point
        
        # 占用率统一扣除
        for interval_index in intervals_index:
            self.state['Node']['intervals']['usage'][interval_index]-=task_usage

            if not self.state['Link']:
                self.state['Link']=()

            # 增加连接
            self.state['Link']=tuple(list(self.state['Link'])+[np.array([action_task_index,interval_index],dtype=np.int32)])
        
        # 在原来的task上面进行记录
        self.state['Node']['tasks']['VM_choice'][action_task_index]=action['VM_choice']
        self.tasks_modified_history.add(action_task_index)
            
        # 计算奖励
        reward=self.reward(intervals_index)

        # 判断是否结束
        done=self.done()
        if done:
            reward+=self.done_reward
        else:
            reward+=self.done_tasks_reward*len(self.tasks_modified_history)

        #记录历史
        self.history_state.append(self.state)
        self.history_reward.append(reward)

        # 返回下一个状态、奖励、是否结束、额外信息
        return self.state,reward,done,"正常完成步骤处理"
    
    def reward(self,intervals_index):
        #查看所经历的每一个间隙，每一个间隙的占用率越高，奖励越高
        action_task_index:np.int32=self.action["task_index"]
        begin_point:np.float32=self.action["begin_point"]
        end_point:np.float32=self.action["begin_point"]+self.state['Node']['tasks']['duration'][action_task_index]
        arrival_point:np.float32=self.state['Node']['tasks']['arrival'][self.action["task_index"]]
        task_usage:np.float32=self.state['Node']['tasks']['usage'][self.action["task_index"]]
        deadline_point:np.float32=self.state['Node']['tasks']['deadline'][self.action["task_index"]]
        reward:np.float32=0.0

        # 对早进行的奖励
        reward+=task_usage*(arrival_point-begin_point)*5
        # 对提前结束的奖励
        # if deadline_point>end_point:
        reward+=task_usage*(deadline_point-end_point)*50
        # else:
            # reward-=0
        # 找到对应占用的间隙
        intervals:collections.OrderedDict=self.state['Node']['intervals']
        # 用间隙的占用率进行奖励，尽量占满间隙
        for interval_index in intervals_index:
            reward+=intervals['usage'][interval_index]*(intervals['end'][interval_index]-intervals['start'][interval_index])
        
        # 计算奖励
        return reward
    
    def done(self):
        # 判断是否结束
        tasks=self.state['Node']['tasks']
        tasks_index=tasks['index']
        #如果所有的任务索引都在链接中，那么就结束
        # print(self.state['Link'])
        task_linked_list=[link[0] for link in self.state['Link']]
        if np.all([task_index in task_linked_list for task_index in tasks_index]):
            return True
        # 如果有任务索引不在链接中，那么就不结束
        return False
    
    def render(self):
        pass

    def close(self):

        pass
    
    def print_history(self):
        for index in range(len(self.history_action)):
            print("第{}步".format(index))
            print("动作：",self.history_action[index])
            print("状态：",self.history_state[index])
            print("奖励：",self.history_reward[index])
            print("")
    
# env=CustomEnv()
# env.stateGenerator(task_node_num=20,interval_node_num_pervm=20)
# env.step({"task_index":0,"VM_choice":0,"begin_point":0.1})
# (state,reward,step)=env.step({"task_index":0,"VM_choice":0,"begin_point":0.1})

# print(state)