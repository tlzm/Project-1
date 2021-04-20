# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:25:28 2018

@author: Admin
"""
from ctypes import *
from scipy.interpolate import interp1d   #插值程序
from Recompose_V1 import MyEngine
from RL_brain import DoubleDQN


import xlwt
import xlrd
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

xk_fn=0
a_fn=1
b_fn=0.125
c_fn=0.08
d_fn=0.005

xk_qmf=0
a_qmf=1
b_qmf=0.125
c_qmf=0.08
d_qmf=0.005


file=xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet=file.add_sheet('aa')

t0= time.clock()

MyEngine = MyEngine()
MyEngine.QD_start()

MEMORY_SIZE = 500  #原值为3000
ACTION_SPACE = 3

sess = tf.Session()


RL = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=2, memory_size=MEMORY_SIZE,

        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())




RL.load_net()

nk = 0
step = 0

cont = 1 # 存储数据用计数器
array = np.array([0], float)
observation_S = array
MyEngine.m_EngineOutPut.sFc_Net = 0.87  # 存储数据时的初值
R = 0
for episode in range(4):
    
#    if episode > 2:
#        MyEngine.Fnc=1300
    
   
    observation = np.array([0,MyEngine.m_EngineOutPut.nFan_R],float) #初始化状态，第一个参数为动作1，第二个参数为NL
    
    for nk in range(1000):
        
        observation_S = np.append(observation_S,observation[0])  # 记录数据
        
        action = RL.choose_action(observation)
        
        S = observation[0]
        
        S_ = S + 0.001 * (float(action)-1)  # 需修改为 S_ =  0.06 * (float(action)-5) 
        
#        if S_ > 0:
#            S_ = 0
#        if S_ < -0.03:
#            S_ = -0.03
#    
#        
        if episode > 0:
            if S_ > 0:
                S_ = 0
            if S_ < -0.035:
                S_ = -0.035
        else:
            if S_ > 0:
                S_ = 0
            if S_ < -0.035:
                S_ = -0.035  
                

        
        for j in range(10):
            
#            if episode > 0:
#                S_ = -0.04
#            else:
#                S_ = -0.035
            
            MyEngine.XunHang(S_)
#            MyEngine.XunHang(0)
            
            if episode > 1:
            
                u_fn = abs(MyEngine.Fnc-MyEngine.m_EngineOutPut.Thrust_Net)       
                y_fn = c_fn*xk_fn + d_fn*u_fn
                xk_fn = a_fn*xk_fn + b_fn*u_fn
                y_fn = y_fn/600
                
                u_qmf = MyEngine.m_EngineInPut.Wf       
                y_qmf = c_qmf*xk_qmf + d_qmf*u_qmf
                xk_qmf = a_qmf*xk_qmf + b_qmf*u_qmf
                y_qmf = y_qmf/10
                
                Score = 100 - y_fn - y_qmf
                
            else:
                y_fn = 0
                y_qmf = 0
                xk_fn = 0
                xk_qmf = 0
                Score = 100 - y_fn - y_qmf           
            
#%%            
        sheet.write(cont,0,MyEngine.m_EngineOutPut.Loc_Fan)
        sheet.write(cont,1,MyEngine.m_EngineOutPut.sFc_Net)
        sheet.write(cont,2,float(MyEngine.Fnc))
        sheet.write(cont,3,float(MyEngine.m_EngineOutPut.Thrust_Net))
        sheet.write(cont,4,float(MyEngine.NLc))
        sheet.write(cont,5,MyEngine.m_EngineOutPut.nFan_C)    
        sheet.write(cont,6,float(MyEngine.pic))
        sheet.write(cont,7,float(MyEngine.m_EngineOutPut.tP25/MyEngine.m_EngineOutPut.tP2))    
        sheet.write(cont,8,MyEngine.m_EngineInPut.Area_Spout)    
        sheet.write(cont,9,MyEngine.m_EngineInPut.Wf)
        sheet.write(cont,10,MyEngine.m_EngineOutPut.nCompress)
        sheet.write(cont,11,S_)
        sheet.write(cont,12,float(action))
        sheet.write(cont,13,float(R))
        sheet.write(cont,14,float(Score))
        sheet.write(cont,15,float(y_fn))
        sheet.write(cont,16,float(y_qmf))
        
        cont = cont + 1
        
        observation_ = np.array([S_,MyEngine.m_EngineOutPut.nFan_R],float)
        
        if MyEngine.m_EngineOutPut.Loc_Fan > 86:
            R = -10*(MyEngine.m_EngineOutPut.Loc_Fan-85)
        else:
            R = (1/float(MyEngine.m_EngineOutPut.sFc_Net))**30
        
        reward = R
        
        RL.store_transition(observation, action, reward, observation_)
    
        if (step > MEMORY_SIZE) :
            RL.learn()
        
        observation = observation_  
        
        step = step + 1

            
plt.plot(observation_S) 
    
#RL.save_net() 

MyEngine.DestroyEngine()

sheet.write(0,0,'SMC')#%%第一列为时间
sheet.write(0,1,'Sfc')
sheet.write(0,2,'Fnc')
sheet.write(0,3,'Fn')
sheet.write(0,4,'NLc')
sheet.write(0,5,'NL')
sheet.write(0,6,'Pic')
sheet.write(0,7,'Pi')
sheet.write(0,8,'A8')
sheet.write(0,9,'Wf')
sheet.write(0,10,'NH')
sheet.write(0,11,'S_')
sheet.write(0,12,'action')
sheet.write(0,13,'R')
sheet.write(0,14,'Score_Z')
sheet.write(0,15,'Score_Fn')
sheet.write(0,16,'Score_qmf')
# 
file.save('name.xls')
t1=time.clock()
print("CPU run time is %f" % (t1-t0))
RL.exit_net()
            
