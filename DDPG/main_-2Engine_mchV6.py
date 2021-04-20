# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:59:21 2019

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:10:07 2019

@author: Admin
"""


import random
import math

import tensorflow as tf

import numpy as np
import xlwt
import xlrd

import time

import matplotlib.pyplot as plt

from ddpg_mch3 import DDPG

from ctypes import *
lib = cdll.LoadLibrary("windll.dll")

#file=xlwt.Workbook(encoding='utf-8',style_compression=0)
#sheet=file.add_sheet('aa')


MAX_EPISODES = 40
MAX_EP_STEPS = 2000
LR_A = 0.001    # learning rate for actor 原值为0.001
LR_C = 0.002    # learning rate for critic 原值为0.002
GAMMA = 0.9     # reward discount  原值为0.9
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 15000   # 原值为100000
BATCH_SIZE = 64 # 原值为32
OUTPUT_GRAPH = True


#函数返回值为double 类型
lib.Get_Engine_NH.restype = c_double
lib.Get_Engine_NL.restype = c_double
lib.Get_Engine_NLc.restype = c_double
lib.Get_Engine_NHc.restype = c_double
lib.Get_Engine_Wat.restype = c_double
lib.Get_Engine_Fn.restype = c_double
lib.Get_Engine_Pt4.restype = c_double
lib.Get_Engine_Tt4.restype = c_double

lib.Get_Engine_wg4.restype = c_double
lib.Get_Engine_Tt3.restype = c_double
lib.Get_Engine_Pt3.restype = c_double
lib.Get_Engine_wa3.restype = c_double
lib.Get_Engine_SMC.restype = c_double
lib.Get_Engine_Tt2.restype = c_double
lib.Get_Engine_Pt2.restype = c_double
lib.Get_Engine_wa2.restype = c_double

lib.Get_Engine_SMF.restype = c_double
lib.Get_Engine_Tt5.restype = c_double
lib.Get_Engine_Pt5.restype = c_double
lib.Get_Engine_wg5.restype = c_double
lib.Get_Engine_Tt6.restype = c_double
lib.Get_Engine_Pt6.restype = c_double
lib.Get_Engine_wg6.restype = c_double
lib.Get_Engine_Time.restype = c_double

lib.Saturation.restype = c_double
lib.GetAbsMax.restype = c_double
lib.LineInterp.restype = c_double
#函数返回值为double 类型


NH = 49000
dNH = c_double(0.0)
NL = c_double(25000.0)
dNL = c_double(0.0)

pif = c_double(1.58)
pic = c_double(3.0)
pith = c_double(1.9)
pitl = c_double(1.4)

VFan = c_double(0.05)
VCom = c_double(0.05)
VHtb = c_double(0.05)
VLtb = c_double(0.05)

step = c_double(0.01)

H = c_double(0)
Ma = c_double(0)
# 参数初始化
cont = 1

lib.Init_AeroDynamicFun()
lib.Init_Engine()
lib.Init_Engine_Size()

lib.Init_Engine_Shaft(c_double(NH),dNH,NL,dNL)
lib.Init_Engine_Pi(pif,pic,pith,pitl)

lib.Init_Volume_Corrector(VFan,VCom,VHtb,VLtb)
lib.Set_Engine_Step(step)


s_dim = 4 # 状态个数
a_dim = 1

a_bound = np.array([1],float)#原值为1

ddpg = DDPG(a_dim, s_dim, a_bound)

#ddpg.load_net()

var = 0.5  # control exploration  var = 0.5
var1 = 0.5

a = 0
action_1 = 0
action = 0
r = 0
NH_gy = 0

observation_S = np.array([0.0],float)
observation_S1 = np.array([0.0],float)
observation_u = np.array([0.0],float)
observation_a = np.array([0.0],float)

t1 = time.time()
wf = 0.02
NH_z = 4.6

a_lb=0.9672
b_lb=0.1796
c_lb=0.1796
d_lb=0.01639
x_lb=0

e_k_4= 0
e_k_3 = 0
e_k_2 = 0
e_k_1 = 0
e_a = 0

flag = 0
ZJBL = 0

for i in range(MAX_EPISODES): 
    
    if (i < 20) :
        NH_z = 4.4
    elif (i < 40):
        NH_z = 4.55
    else:
        NH_z = 4.85
    
    s = np.array([0,0,0,0],float)
    
    ep_reward = 0
    

    for j in range(MAX_EP_STEPS):
        
        e_k = float(NH_z) - float(NH_gy)        

        s = np.array([e_k,e_k_1,ZJBL,NH_gy],float)   
        
        a = ddpg.choose_action(s) 
        
        if cont % 100 == 0:
            var1 = var
        else:
            var1 = 0                 

        action = np.clip(np.random.normal(a, var1), -1,1)    # add randomness to action selection for exploration
                
        action =  float(action)
        
        ZJBL = ZJBL + action
        
        if ZJBL > 5:
            ZJBL = 5
        if ZJBL < 0.1:
            ZJBL = 0.1       
     
        wf = ZJBL*6/490 + (0.08-3/49)
        
#        wf_lb = c_lb*x_lb + d_lb*wf        
#        x_lb = a_lb*x_lb + b_lb*wf
        
        lib.Set_Engine_Input(c_double(wf),H,Ma)
        
        for k in range(4):
            lib.Engine_Calculate()
            lib.Updata_Engine_Time()
            
        NH = lib.Get_Engine_NH()
        NH_gy = NH / 10000
     

        e_k = NH_z - NH_gy
        e_a = (e_k - e_k_1) /0.01
        
        
        r = - (abs(e_k)+abs(e_k_1))
        
        s_ = np.array([e_k,e_k_1,ZJBL,NH_gy],float)
    
        ddpg.store_transition(s, action, r/10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r

        
        observation_S = np.append(observation_S,NH_gy)
        observation_S1 = np.append(observation_S1,NH_z)
        observation_u = np.append(observation_u,wf)
        observation_a = np.append(observation_a,action) 
        
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %.2f' % r, 'Explore: %.2f' % var,'wf: %.3f'% wf,'NH_gy: %.2f'% NH_gy,'action: %.1f'% action,'e_k: %.1f'% e_k,'e_a: %.1f'% e_a)
            break
        
        e_k_4 = e_k_3
        e_k_3 = e_k_2
        e_k_2 = e_k_1
        e_k_1 = e_k
        action_1 = action
        
#        sheet.write(cont,0,NH)
#        sheet.write(cont,1,float(a))
#        sheet.write(cont,2,e_k)
#        sheet.write(cont,3,e_a) 
#        sheet.write(cont,4,NH_z)
#        sheet.write(cont,5,r)
#        sheet.write(cont,6,wf)
#        sheet.write(cont,7,float(action))        
        
        cont = cont + 1
        
#        if r < 0.001:
#            break

#sheet.write(0,0,'NH')#%%第一列为时间
#sheet.write(0,1,'a')
#sheet.write(0,2,'e_k')
#sheet.write(0,3,'e_a')
#sheet.write(0,4,'NH_z')
#sheet.write(0,5,'r')
#sheet.write(0,6,'wf')
#sheet.write(0,7,'action')        
        
plt.subplot(311)

plt.plot(observation_S)
plt.plot(observation_S1)
plt.ylabel("NH")

plt.subplot(312)
plt.plot(observation_u)
plt.ylabel("wf")

plt.subplot(313)
plt.plot(observation_a)
plt.ylabel("action")

print('Running time: ', time.time() - t1)

#file.save('mch.xls')

#ddpg.save_net()

ddpg.exit_net()