
#TODO 精度是否有问题？
#TODO 在线学习（研究点二）
#TODO 将研究点一得到的数据用于研究点二的实验
#TODO 将接收耗费的时间纳入考虑
#TODO 在使用delayed策略进行任务调度的时候，可能可以通过保存调度结果减少运算
#TODO 是否还有更好的节省调度资源的方法
import pandas as pd
import os
import winsound
from PPO import PPO
import PPO_prime_run
from CustomEnv import CustomEnv
import collections
import numpy as np
import copy
import gymnasium as gym
#-----------------------------------------------------------
duration = 2000  # millisecond
freq = 600  # Hz

MIN_COMPARE=1e-6

#根据20230712TaskMatrix.ipynb确定
LAM=20
MU=12
PM_COUNT=4
inf_replace=30000
#文件名头
result_head:str=f'LAM{LAM}_MU{MU}_PMCOUNT{PM_COUNT}_'
task_head:str=f'LAM{LAM}_MU{MU}_'

task_columns=['arrival_point','handling_time','usage(%)','max_wait_time','start_point']
'''['arrival_point','handling_time','usage(%)','max_wait_time','start_point']'''
server_columns=['point','usage(%)']
'''['point','usage(%)']'''
request_queue=pd.DataFrame(columns=task_columns)
'''task_columns=['arrival_point','handling_time','usage(%)','max_wait_time','start_point']'''
delayed_queue=pd.DataFrame(columns=task_columns)
'''task_columns=['arrival_point','handling_time','usage(%)','max_wait_time','start_point']'''

time:float=0.0

path=f"./{task_head}20230712TaskMatrix.json"#在文件20230712TaskMatrix.ipynb第65格中创建
task_matrix:pd.DataFrame=pd.read_json(path)
timeout=pd.DataFrame(columns=task_columns)
'''
记录超时的任务\n
task_columns=['arrival_point','handling_time','usage(%)','max_wait_time','start_point']
'''
#task_matrix具体结构如下
'''
df = pd.DataFrame(columns=('Task_ID',
                           'Arrival_time',
                           'Receive_time_span',
                           'Start_computing_time',
                           'Process_time_span',
                           'Finish'
                           'Max_wait_time_span',
                           'DeadLine',
                           'Cpu_utilization'))
'''

# interval_priority=dict(by=['max_usage(%)','temp_interval_time','begin_point'],ascending=[False,True,True])#new
# interval_priority=dict(by=['begin_point','max_usage(%)'],ascending=[True,False])#MaxUtil
# delayed_task_priority=
# dispatch_file_name="new"
# dispatch_file_name="MaxUtil"
# task_priority_file_name="deadline"
# task_priority_file_name="beginpoint"
interval_priority_choice:int=3
'''0是new，1是MaxUtil，2是Random，3是近端策略优化'''
task_priority_choice:int=1
'''0是deadline，1是arrivalpoint，2是推迟调度；该项仅对推迟调度（包括延期）的任务有效'''



random_status:bool=False
'''random因为不涉及排序优先级，所以单独使语句，需要单独写'''
if interval_priority_choice==2 or interval_priority_choice==3:
    random_status=True

delay_request_status:bool=False
if task_priority_choice==2:
    delay_request_status=True
#是否推迟再判断，不仅对new算法有效，因为maxutil有可能也要因为拥挤推迟。但是只对deadline优先的有效
#TODO 缺乏一个预约的机制，可能好几个任务都预约到的是同一个间隙，可能需要另外导入一个变量就是predict_time_table
#TODO 可能可以使用深度学习的方法去决定间隙？具体还没有想法


#测试使用lambda 25
# model_load_path='./TaskIntervalEnv/prime_seed_0_0509_04-59-58/PPO_TaskIntervalEnv_0_3000000_0509_10-47-42.pth'

# TaskIntervalEnv/prime_seed_0_0509_11-24-41/PPO_TaskIntervalEnv_0_500000_0509_13-47-12.pth
model_load_path='./TaskIntervalEnv/prime_seed_0_0509_14-56-48/PPO_TaskIntervalEnv_0_500000_0509_17-24-21.pth'#6298 #可以查修改的历史记录
# model_load_path='TaskIntervalEnv\prime_seed_0_0509_23-35-14\PPO_TaskIntervalEnv_0_500000_0510_02-06-42.pth'#6650
# model_load_path='TaskIntervalEnv\prime_seed_0_0509_14-56-48\PPO_TaskIntervalEnv_0_2000000_0509_22-12-58.pth'#6345
# model_load_path='TaskIntervalEnv\prime_seed_0_0510_11-43-58\PPO_TaskIntervalEnv_0_500000_0510_15-22-04.pth'#失败
# model_load_path='TaskIntervalEnv\prime_seed_0_0510_11-43-58\PPO_TaskIntervalEnv_0_1000000_0510_18-55-23.pth'#失败
# model_load_path='TaskIntervalEnv\prime_seed_0_0510_19-29-47\PPO_TaskIntervalEnv_0_500000_0510_21-08-04.pth'#失败

interval_priority_list:list=[{'interval_priority':dict(by=['max_usage(%)','temp_interval_time','begin_point'],ascending=[False,True,True]),'dispatch_file_name':"new"},{'interval_priority':dict(by=['begin_point','max_usage(%)'],ascending=[True,False]),'dispatch_file_name':"MaxUtil"},{'interval_priority':dict(),'dispatch_file_name':"random"},{'interval_priority':dict(),'dispatch_file_name':"PPO"}]

interval_priority:dict=interval_priority_list[interval_priority_choice]['interval_priority']
'''
interval_priority=interval_priority_list[interval_priority_choice]['interval_priority']\n
对于interval_priority_choice：0是new，1是MaxUtil，2是Random
'''

dispatch_file_name:str=interval_priority_list[interval_priority_choice]['dispatch_file_name']
'''
dispatch_file_name=interval_priority_list[interval_priority_choice]['dispatch_file_name']\n
对于interval_priority_choice：0是new，1是MaxUtil，2是Random
'''
#TODO 是否可以让temp_interval_time最优先？

task_priority_list:list=[{'task_priority':dict(by='temp_deadline',ascending=True),'task_priority_file_name':"deadline"},{'task_priority':dict(by='arrival_point',ascending=True),'task_priority_file_name':"arrivalpoint"},{'task_priority':dict(by='temp_deadline',ascending=True),'task_priority_file_name':"delayed"}]

task_priority:dict=task_priority_list[task_priority_choice]['task_priority']
'''
task_priority=task_priority_list[task_priority_choice]['task_priority']\n
对于task_priority_choice：0是deadline，1是arrivalpoint，2是推迟调度；该项仅对推迟调度（包括延期）的任务有效
'''

task_priority_file_name:str=task_priority_list[task_priority_choice]['task_priority_file_name']
'''
task_priority_file_name=task_priority_list[task_priority_choice]['task_priority_file_name']\n
对于task_priority_choice：0是deadline，1是arrivalpoint，2是推迟调度；该项仅对推迟调度（包括延期）的任务有效
'''


B_things_list=[]
C_things_list=[]
'''每个循环结束都会变成空的'''
new_task_flag=0
'''当轮剩余要处理的新任务数量'''
task_complete_flag=0

#-----------------------------------------------------------------------------------------
PPO_model_test_on=True
PPO_on=False
#关于PPO模型的在线或测试，对类的初始化
if interval_priority_choice==3:
    PPO_on=True

# if PPO_model_test_on and PPO_on:
#     ppo_agent = PPO( lr_actor, lr_critic, gamma, K_epochs, eps_clip, continuous_action_dim,discrete_action_dim, action_std)

#     ppo_agent.load(model_load_path)



#-----------------------
class Server():
    def __init__(self,server_index) -> None:
        self.server_index=server_index
        self.time_table=pd.DataFrame(data=[[0.0,0.0]],columns=server_columns)
        '''
        记录一个服务器的时刻和占用率情况。对于每一行，包含时刻和占用率。占用率指的是该行时刻至下一行时刻之间的占用率\n
        server_columns=['point','usage(%)']
        '''
        # print('构造函数中')
        # print(self.time_table)
        self.task_table=pd.DataFrame(columns=task_columns)
        '''
        记录一个服务器运行的任务情况\n
        task_columns=['arrival_point','handling_time','usage(%)','max_wait_time','start_point']
        '''
        # server_columns=['point','usage(%)']
        # task_columns=['arrival_point','handling_time','usage(%)','max_wait_time','start_point']
        #这两个列表均是记录所有的任务和时间情况，不会进行删除
        #初始化一下第一项
        
    def task_reconduct(self):
        return#TODOinthefuture

#创建指定个数个服务器
PM_list=[Server(server_index=index) for index in range(PM_COUNT)]

class Things():
    def __init__(self,time):
        self.time=time

class B(Things):
    def __init__(self, time):
        super().__init__(time)
    def run(self):
        return

class C(Things):
    def __init__(self,time):
        super().__init__(time)
        self.interval_column_list=['begin_point','end_point','PM_index','max_usage(%)']
        self.intervals=pd.DataFrame(columns=self.interval_column_list)#创建时间间隙表
        '''
        interval_column_list=['begin_point','end_point','PM_index','max_usage(%)']
        '''
        self.interval_choice=pd.Series(index=self.interval_column_list)#在时间间隙表中选择最合适的时间间隙
        '''
        interval_column_list=['begin_point','end_point','PM_index','max_usage(%)']
        '''

    def condition(self)->bool:
        return False

    def run(self):
        return
    
    def delay_request(self,task):
        global delayed_queue
        delayed_queue=pd.concat([delayed_queue,task.to_frame().T],ignore_index=True)
        # os.system('pause')
        
        print(delayed_queue)

    def find_intervals(self,task:pd.Series,infty_limit=True)->None:
        '''
        对于intervals每一行:self.interval_column_list=['begin_point','end_point','PM_index','max_usage(%)']
        不包含选择最小的那个间隔
        '''
        global PM_list
        global time
        global timeout
        global interval_priority
        global random_status
        global inf_replace
        global interval_priority_choice
        print("任务为")
        print(task.to_frame().T)
        time_frame=task['handling_time']+task['max_wait_time']-(time-task['arrival_point'])

        if time_frame<task['handling_time']:#针对超时事件的处理
            time_frame=task['handling_time']
            # timeout=pd.merge(timeout,task.to_frame().T,how='outer')
            
            print("正在处理超时事件的间隙查找问题")
        print(f"正在寻找间隙，寻找的时间长度为{time_frame}")
        for PM_index,server in enumerate(PM_list):
            # print(f"正在查询服务器{PM_index}")
            server:Server
            #查询时间表
            time_table=server.time_table.copy()#复制时间表
            if infty_limit:
                time_table:pd.DataFrame=time_table.loc[time_table['point']<time+time_frame]
            time_table:pd.DataFrame=time_table.sort_values(by='point',ascending=True,ignore_index=True)#排序
            # print(f"第一次筛选时间节点列表如下，time_table['point']<time+time_frame")
            # print(time_table)
            #不管有无重合都要加入末尾元素。在第一次筛选的时候就不考虑末尾相同的是因为利用率不具有参考性
            #因为是前闭后开，所以统一考虑头部不考虑尾部。如果遇到最后一条OK那么空余时间可以被自动设置到间隙末尾。如果不是最后一条的话，间隙也可以被设置到下一个时刻。
            #这里采用加入虚拟的最后一条来达到时刻统计上的统一
            #注意end_point需要在头部处理完成之后进行，防止出现空的情况

            # print("组合上截止时间点的时间表如下：")
            # print(time_table)
            if time_table.loc[time_table['point']==time].empty:#如果没有先头元素的话就加上一个先头元素
                before_time_table:pd.DataFrame=time_table.loc[time_table['point']<time]
                before_point:pd.Series
                if before_time_table.empty:
                    before_point:pd.Series=pd.Series(data=[0.0,0.0],index=server_columns)
                else:
                    before_point:pd.Series=before_time_table.iloc[-1].copy()#这里默认,排序是OK的
                before_point['point']=time
                # print("根据判断，需要起始时间点，起始时间点如下")
                # print(before_point)
                time_table:pd.DataFrame=pd.concat([time_table,before_point.to_frame().T],ignore_index=True)
            time_table:pd.DataFrame=time_table.loc[time_table['point']>=time].sort_values(by='point',ascending=True,ignore_index=True)

            # print(f"处理之后的服务器时间表，time_table['point']>=time，如下：")
            # print(time_table)

            #初始化变量
            interval=pd.Series(index=self.interval_column_list)#这里的index指的是行标签
            #需要注意的是，注意搜索的范围和time_frame有关，不要搜索过多的B事件
            #也有一种可能是，time_frame本身超出了B事件最长的范围
            time_table_rows_num=time_table.shape[0]#行数（不加入最后一个节点）
            # acceptable_indexes:list=time_table[(time_table['usage(%)']+task['usage(%)']<=100)].index.tolist()
            # print("acceptable_indexes:")
            # print(acceptable_indexes)
            # if not acceptable_indexes:#都没有利用率满足要求
            #     continue
            end_point:pd.Series=time_table.iloc[-1].copy()#注意深拷贝和浅拷贝的问题
            if infty_limit:
                end_point['point']=time+time_frame
            else:
                end_point['point']=inf_replace
            # end_point['usage(%)']=100
            #将截止时间放在时刻表后面，这样可以不受到最后一个事件发生时间的影响
            # print("截止时间点")
            # print(end_point)
            time_table:pd.DataFrame=pd.concat([time_table,end_point.to_frame().T],axis=0,ignore_index=True)
            # print("time_table:")
            # print(time_table)

            
            #为了防止遍历两遍，使用统一的方法进行
            if interval_priority_choice!=3:
                begin_index:int=-1
                for index,row in time_table.iterrows():
                    if index==time_table_rows_num or row['usage(%)']+task['usage(%)']>100:#如果是最后一个元素或者是超过了100%的利用率
                        if begin_index!=-1 and time_table.loc[index,'point']-time_table.loc[begin_index,'point']>=task['handling_time']-MIN_COMPARE:
                            self.deal_interval(time_table=time_table,interval=interval,PM_index=PM_index,begin_index=begin_index,end_index=index)
                        begin_index=-1
                    elif begin_index==-1:
                        begin_index=index
            else:
                for index,row in time_table.iterrows():
                    if index==0:
                        continue
                    self.deal_interval(time_table=time_table,interval=interval,PM_index=PM_index,begin_index=index-1,end_index=index)
                    

            
        if interval_priority_choice!=3:
            self.intervals['temp_interval_time']=self.intervals['end_point']-self.intervals['begin_point']
            if random_status:
                self.intervals:pd.DataFrame=self.intervals.sample(frac=1)#全部打乱
            else:
                self.intervals:pd.DataFrame=self.intervals.sort_values(**interval_priority).drop(columns=['temp_interval_time'])#优先按照间隔排序，其次按照开始时刻排序#TODO
                # 应该改成优先按照占用率排序，其次是间隔，最后是开始时刻，所以可能需要在intervals中加入新的字段？  已经加入新字段max_usage(%)
            # print("所有的可能的间隙如下：(不考虑时间长度)")
            # print(self.intervals)
        return
    
    def deal_interval(self,time_table:pd.DataFrame,interval:pd.Series,PM_index:int,begin_index:int,end_index:int):
        interval['begin_point']=time_table.loc[begin_index,'point']
        interval['end_point']=time_table.loc[end_index,'point']
        interval['max_usage(%)']=time_table['usage(%)'].iloc[begin_index:end_index].max()#不包含end_index#注意loc是包含的
        interval['PM_index']=PM_index
        self.intervals=pd.concat([self.intervals,interval.to_frame().T],ignore_index=True)
        return


class B1(B):
    '''
    B1事件，任务到达并进入等待序列
    '''
    def __init__(self,task:pd.Series,server_utilization:list=[[100.0] for i in range(PM_COUNT)]):
        super(B1,self).__init__(time=float(task['arrival_point']))
        self.task=task

    def run(self):
        '''
        将任务加入请求队列
        '''
        global request_queue
        global new_task_flag
        request_queue=pd.concat([request_queue,self.task.to_frame().T],ignore_index=True)
        new_task_flag=new_task_flag+1
        print(f"正在运行B1事件，目前new_task_flag={new_task_flag}")
        return

class B2(B):#考虑简洁，就不另外传入时间，而是算一下先
    '''
    任务完成，虚拟机减少负载
    '''
    def __init__(self,task):
        super(B2,self).__init__(time=float(task['start_point']+task['handling_time']))
        self.task:pd.Series=task
    
    def run(self):
        '''
        虚拟机自然而然会减小负载,不需要额外运行什么
        '''
        print(f"目前正在运行B2事件，将直接返回")
        print("结束的任务为：")
        print(self.task.to_frame().T)
        return
    
class C1(C):
    '''
    C1事件，在有任务到达的情况下执行，
    检测是否服务器全忙，如果服务器满足塞的条件则将一个任务从队列中取出，分配给服务器按照计划进行工作，如果服务器全忙则保持在等待队列中。注意不是一下子处理所有的任务，是一个个处理任务
    '''
    def __init__(self,time):
        super().__init__(time)

    def condition(self) -> bool:
        global new_task_flag
        if interval_priority_choice==3:
            return False
        if new_task_flag>0:
            new_task_flag=new_task_flag-1
            return True
        else:
            return False
    




    def resolve_request(self,task:pd.Series)->None:
        '''
        在已经知道相对于这个请求最合适的间隙的情况的情况下处理这个请求
        '''
        #不知道关于调用的函数的复杂度是不是可以降低一些,是不是有更好一些的函数可以处理这样的需求

        #向对应的服务器加入任务

        global time
        global PM_list
        global timeout
        
        #统计延误的任务
        delayed:bool=False
        if task['start_point']<-0.5:
            delayed=True
        task['start_point']=self.interval_choice['begin_point']
        if delayed:
            timeout=pd.concat([timeout,task.to_frame().T],ignore_index=True)
        server:Server=PM_list[int(self.interval_choice['PM_index'])]

        #更新server的time_table
        #如果开始时刻没有对应的时间节点
        # print("正在面对选定的间隔对服务器的时间表进行处理")
        # print(f"server_index:{server.server_index}")
        # print("server.time_table")
        # print(server.time_table)
        #设置任务的开始时间是间隙的开始时间
        if server.time_table.loc[server.time_table['point']==task['start_point']].empty:
            before_time_table:pd.DataFrame=server.time_table.loc[server.time_table['point']<task['start_point']]
            last_usage=0.0
            if not before_time_table.empty:
                last_usage:float=before_time_table.sort_values(by='point',ascending=False,ignore_index=True).loc[0,'usage(%)']#找到上一个时间节点的利用率
            #加入时间节点
            time_node=pd.Series(data=[task['start_point'],last_usage],index=server_columns)#最后利用率统一加
            server.time_table=pd.concat([server.time_table,time_node.to_frame().T])
            #注意需要更新一下时间列表,因为可能出现任务刚刚好在两个时间节点之间的情况
            server.time_table=server.time_table.sort_values(by='point',ascending=True,ignore_index=True)
        # print(server.time_table)
        #如果结束时刻没有对应的时间节点
        if server.time_table.loc[server.time_table['point']==task['start_point']+task['handling_time']].empty:
            #则需要加入时间节点
            last_usage:float=server.time_table.loc[server.time_table['point']<task['start_point']+task['handling_time']].sort_values(by='point',ascending=False,ignore_index=True).loc[0,'usage(%)']#找到上一个时间节点的利用率
            #加入时间节点
            time_node=pd.Series(data=[task['start_point']+task['handling_time'],last_usage],index=server_columns)
            server.time_table=pd.concat([server.time_table,time_node.to_frame().T])
            server.time_table=server.time_table.sort_values(by='point',ascending=True,ignore_index=True)
        # print(server.time_table)
        #设置筛选条件
        mask=((server.time_table['point']>=task['start_point'])&(server.time_table['point']<task['start_point']+task['handling_time']))
        server.time_table.loc[mask,'usage(%)']=server.time_table.loc[mask,'usage(%)']+task['usage(%)']#对符合条件的time_table里的服务器已有资源加上一部分

        
        print("正在面对选定的间隔对服务器的时间表进行处理")
        print(f"server_index:{server.server_index}")
        print("server.time_table")
        print(server.time_table.iloc[-10:])
        #更新服务器的task-table
        server.task_table=pd.concat([server.task_table,task.to_frame().T],ignore_index=True)

        #更新B事件
        B_things_list.append(B2(task))
        print("加入事件B2，对应的任务详细情况如下")
        print(task.to_frame().T)

        return



    def run(self):
        '''
        检测是否服务器全忙，如果服务器满足塞的条件则将任务从队列中取出，分配给服务器按照计划进行工作，如果服务器全忙则保持在等待队列中
        '''
        global PMCOUNT
        global time
        global request_queue
        global delayed_queue
        global freq
        global duration
        global task_priority
        #对任务列表内的任务进行排序，优先指标：最长等待时间+计算时间-（现在时间-到达时间），也就是期限最早的.这里使用临时列进行自定义排序#TODO
        request_queue['temp_deadline']=request_queue['max_wait_time']+request_queue['handling_time']-time+request_queue['arrival_point']
        for index,request in request_queue.iterrows():
            request:pd.Series
            if request['temp_deadline']>request['handling_time']:
                request_queue.loc[index,'temp_deadline']=request['handling_time']
        request_queue=request_queue.sort_values(**task_priority,ignore_index=True).drop(columns=['temp_deadline'])
        task:pd.Series=request_queue.iloc[0]#截取中其中的一个任务
        #注意需要更新一下request_queue,因为需要进行C的状态判断，所以必须删除这一条，所以说需要一个delayed_queue
        request_queue.drop([0],axis=0,inplace=True)
        #找一下可以用的间隙
        self.find_intervals(task=task)
        print(f"正在进行C1事件，目前的可用间隙如下")
        print(self.intervals)
        
        #将任务安排到间隙，需要更新服务器和更新事件
        if self.intervals.empty:#找不到间隙的话就将任务加入到延迟队列中
            task['start_point']=-1.0
            self.delay_request(task)
            print("找不到间隙，将任务加入延迟列表：")
            # winsound.Beep(freq, duration)
            # os.system("pause")
            return
        self.interval_choice:pd.Series=self.intervals.iloc[0]
        print("选择的间隙如下：")
        print(self.interval_choice.to_frame().T)
        #如果可以找得到间隙的话
        if delay_request_status and time<self.interval_choice['begin_point']-MIN_COMPARE:
            self.delay_request(task)
            print("延迟任务，之后再定间隙")
            return
        self.resolve_request(task)
        return

class C2(C):
    #用于和PPO的对接
    def __init__(self,time):
        super().__init__(time)

    def condition(self) -> bool:
        global new_task_flag
        global PPO_on 
        global PPO_model_test_on
        if PPO_on and PPO_model_test_on and new_task_flag>0:
            new_task_flag=0
            return True
        else: 
            return False
    
    def Dataframe2OrderedDict(self)->collections.OrderedDict:
        '''
        将dataframe转换成OrderedDict
        '''
        global time
        global delayed_queue
        global request_queue
        intervals_dict=collections.OrderedDict()
        intervals_num=self.intervals.shape[0]
        intervals_dict['index']=np.arange(intervals_num)
        intervals_dict['start']=np.array(self.intervals['begin_point']-time)
        intervals_dict['end']=np.array(self.intervals['end_point']-time)
        intervals_dict['VM']=np.array(self.intervals['PM_index'])
        intervals_dict['usage']=1-np.array(self.intervals['max_usage(%)']/100)#注意这里是1-，因为env考虑的是剩余的资源

        tasks_dict=collections.OrderedDict()
        #可以考虑延迟策略，之后再考虑
        # if delay_request_status:
        #     delayed_queue=request_queue.copy()
        #     delayed_queue=delayed_queue.loc[delayed_queue['start_point']>time]
        #     request_queue=request_queue.loc[request_queue['start_point']<=time]
        
        tasks_num=request_queue.shape[0]
        

        tasks_dict['index']=np.arange(tasks_num)
        tasks_dict['arrival']=np.array(request_queue['arrival_point']-time)
        tasks_dict['duration']=np.array(request_queue['handling_time'])
        tasks_dict['usage']=np.array(request_queue['usage(%)']/100)
        tasks_dict['deadline']=np.array(request_queue['max_wait_time']+request_queue['handling_time']-time+request_queue['arrival_point'])
        tasks_dict['VM_choice']=np.full(tasks_num,-1)

        init_state=collections.OrderedDict({'Node':collections.OrderedDict({'intervals':intervals_dict,'tasks':tasks_dict}),'Link':()})
        return init_state
    
    def OrderedDict2Dataframe(self,ret_state:collections.OrderedDict,action_list:list)->pd.DataFrame:
        '''
        将OrderedDict转换成dataframe
        '''
        global time
        ret_state:dict=ret_state['Node']
        intervals=pd.DataFrame(data={'begin_point':ret_state['intervals']['start']+time,'end_point':ret_state['intervals']['end']+time,'PM_index':ret_state['intervals']['VM'],'max_usage(%)':100-ret_state['intervals']['usage']*100})
        tasks=pd.DataFrame(data={'arrival_point':ret_state['tasks']['arrival']+time,'handling_time':ret_state['tasks']['duration'],'usage(%)':ret_state['tasks']['usage']*100,'max_wait_time':ret_state['tasks']['deadline']-ret_state['tasks']['arrival']-ret_state['tasks']['duration'],'start_point':0.0,'PM_index':ret_state['tasks']['VM_choice']})
        for action in action_list:
            action:collections.OrderedDict
            index=action['task_index']
            tasks.loc[index,'PM_index']=action['VM_choice']
            tasks.loc[index,'start_point']=action['begin_point']+time
        return intervals,tasks

    def run(self):
        global time
        global task_columns
        global request_queue
        global delayed_queue
        global PM_list
        global timeout
        global interval_priority
        global inf_replace
        global model_load_path
        # global random_status
        # global inf_replace
        # global task_priority


        #转成state(OrderedDict)的形式
        self.find_intervals(task=pd.Series(data=[time,0.0,0.0,inf_replace,0.0],index=task_columns),infty_limit=False)
        '''
        对于intervals每一行:self.interval_column_list=['begin_point','end_point','PM_index','max_usage(%)']
        
        '''

        init_state=self.Dataframe2OrderedDict()
        intervals_num=self.intervals.shape[0]
        tasks_num=request_queue.shape[0]
        
        #创建env
        self.env=CustomEnv(tasks_num,intervals_num,PM_COUNT)
        
        self.env.state=init_state
        self.env.initial_state=init_state
        

        #调用PPO模型进行使用
        K_epochs = 10               # update policy for K epochs in one PPO update 计算一次损失
        eps_clip = 0.2          # clip parameter for PPO
        gamma = 0.99            # discount factor
        lr_actor = 0.0       # learning rate for actor network
        lr_critic = 0.0      # learning rate for critic network
        action_std = 1e-5 

        discrete_action_dim:dict=dict()
        continuous_action_dim = 0
        
        
        # 对ppo_agent进行一个修正，使得可以对接上环境的输出
        for space_name, sub_space in self.env.action_space.spaces.items():
            if isinstance(sub_space, gym.spaces.Discrete):
                #注意这边一定要以gymnasium开头，不然会判断为False
                discrete_action_dim[space_name]=gym.spaces.flatdim(sub_space)
            elif isinstance(sub_space, gym.spaces.Box):
                continuous_action_dim += gym.spaces.flatdim(sub_space)

        # interval_node_num=int(np.random.randint(8,50))
        # task_node_num=int(np.random.randint(1,20))
        # # vm_num=int(np.random.randint(2,10))
        # vm_num=4
        # self.env.reset(task_node_num=task_node_num,interval_node_num=interval_node_num,vm_num=vm_num)

        self.ppo_agent = PPO( lr_actor, lr_critic, gamma, K_epochs, eps_clip, continuous_action_dim,discrete_action_dim, action_std)

        self.ppo_agent.load(model_load_path)
        
        # self.env.reset(tasks_num,intervals_num,PM_COUNT)
        self.ppo_agent.policy.eval()
        self.ppo_agent.policy_old.eval()


        ret_state,action_list=self.test()
        #将ret_state转换成dataframe
        intervals,tasks=self.OrderedDict2Dataframe(ret_state,action_list)
        intervals:pd.DataFrame
        tasks:pd.DataFrame

        #将intervals和tasks加入到相应的列表中
        #对应不同的PM将intervals转成 time_table
        for PM_index,server in enumerate(PM_list):
            PM_new_data=intervals.loc[intervals['PM_index']==PM_index].drop(columns=['PM_index','end_point']).sort_values(by='begin_point',ascending=True,ignore_index=True)
            PM_new_data.rename(columns={'begin_point':'point','max_usage(%)':'usage(%)'},inplace=True)
            
            PM_old_dict=dict(zip(server.time_table['point'],server.time_table['usage(%)']))
            for index,row in PM_new_data.iterrows():
                PM_old_dict[row['point']]=row['usage(%)']
            server.time_table=pd.DataFrame(data=PM_old_dict.items(),columns=server_columns)
            server.time_table=server.time_table.sort_values(by='point',ascending=True,ignore_index=True)
            
        
        
        for index,task in tasks.iterrows():
            task:pd.Series
            #将没有处理的tasks加入到delayed_queue中
            if task['PM_index']<-0.5:
                delayed_queue=pd.concat([delayed_queue,task.drop(['PM_index'],inplace=False).to_frame().T],ignore_index=True)

            else:#对于没有延迟的任务
                #将任务放在task_table中
                server:Server=PM_list[int(task['PM_index'])]
                server.task_table=pd.concat([server.task_table,task.drop(['PM_index'],inplace=False).to_frame().T],ignore_index=True)
                #加入B事件
                B_things_list.append(B2(task))
                #将超时的任务加入到timeout中
                if task['start_point']-task['arrival_point']>task['max_wait_time']:
                    timeout=pd.concat([timeout,task.drop(['PM_index'],inplace=False).to_frame().T],ignore_index=True)

    def test(self):
    
        # global time_step, i_episode, print_running_reward, print_running_episodes, print_running_step, print_done_num, print_episode_num, log_running_reward, log_running_episodes, log_running_step
        global time

        state=copy.deepcopy(self.env.state)
        state.pop('Link')
        current_ep_reward = 0
        current_ep_state_value=0
        output_str=""
        max_ep_len=int(self.env.task_node_num*20)
        action_list=[]
        for t in range(1, max_ep_len+1):
            #更新state
            state=copy.deepcopy(self.env.state)
            # select action with policy
            action:collections.OrderedDict = self.ppo_agent.select_action(state)
            
            state_ret, reward, done, output_str = self.env.step(action)
            if output_str=="正常完成步骤处理":
                action_list.append(action)
            state_value=self.ppo_agent.critic_forward().item()
            reward = np.float32(reward)
            done: bool = done
            current_ep_reward += reward
            current_ep_state_value+=state_value
            if done:
                break
        print(f"{done}. finished after {t} timesteps.\t Tasks num: {self.env.task_node_num} \tOutput: {output_str}. \tReward: {current_ep_reward} \tState_value: {current_ep_state_value}")
        self.env.close()
        return state_ret,action_list
            
class C3(C2):
    #尝试进行在线学习，使用统一的PPO进行学习
    def __init__(self, time):
        super().__init__(time)

    def condition(self)->bool:
        global new_task_flag
        global PPO_on
        global PPO_model_test_on
        if PPO_on and not PPO_model_test_on and new_task_flag>0:
            new_task_flag=0
            return True
        else:
            return False
        
    


C_class_list=[C3,C2,C1]

if __name__=='__main__':
    #注意事件列表中只存B事件
    #列出所有目前已知的B事件并加入列表
    for index,detailed_task in task_matrix.iterrows():
        #暂时不考虑接收用时长
        task_dict:dict={'arrival_point':detailed_task['Arrival_time'],
                        'handling_time':detailed_task['Process_time_span'],
                        'usage(%)':detailed_task['Cpu_utilization']*100,
                        'max_wait_time':detailed_task['Max_wait_time_span'],
                        'start_point':0.0}#注意一定要加列表的括号（作为一个列表）
        # 目前暂时没有确定要不要考虑进接收时长所以说先使用其他方法来代替
        task=pd.Series(task_dict)
        B_things_list.append(B1(task=task))
        round_index:int=0
    while True:
        if B_things_list==[]:
            print("所有任务处理结束")
            break
        #更新B事件列表
        B_things_list.sort(key=lambda B_thing : B_thing.time)
        #找到距离目前最近的B事件，更新时间节点
        time=B_things_list[0].time
        print(f'round_index:{round_index},time:{time}')
        print(f'B事件数量还剩：{len(B_things_list)}')
        #找到到时间的B事件
        B_allowed_things_list=[B_thing for B_thing in B_things_list if B_thing.time==time]
        for B_thing in B_allowed_things_list:
            print(f"运行B事件，类型是{type(B_thing)}")
            B_thing.run()
            B_things_list.remove(B_thing)#从B事件列表中清除
            print("request_queue")
            print(request_queue)
        #逐条查找符合要求的C事件
        while True:
            for C_class in C_class_list:
                C_thing=C_class(time=time)#实例化
                if C_thing.condition():
                    print(f"加入C事件，类型是{type(C_thing)}")
                    C_things_list.append(C_thing)#加入事件
            if C_things_list==[]:#如果再也找不到符合条件的C事件了的话
                break
            for C_thing in C_things_list:
                print(f"运行C事件，类型是{type(C_thing)}")
                C_thing.run()
            C_things_list=[]
        request_queue=delayed_queue.copy()
        # env=CustomEnv()
        delayed_queue=pd.DataFrame(columns=task_columns)
        new_task_flag=len(request_queue)
        round_index=round_index+1
        # os.system("pause")
        print("---------------------------------------------------------------")
    #记录数据之后用于统计
    for server in PM_list:
        server:Server
        server.time_table.to_json(f"./data/{result_head}{dispatch_file_name}_{task_priority_file_name}_PM{server.server_index}_time_table.json")
        server.task_table.to_json(f"./data/{result_head}{dispatch_file_name}_{task_priority_file_name}_PM{server.server_index}_task_table.json")

    print("timeout")
    print(timeout)
    timeout.to_json(f"./data/{result_head}{dispatch_file_name}_{task_priority_file_name}_timeout.json")
    winsound.Beep(freq, duration)


