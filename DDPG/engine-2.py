
"""
Created on Tue Apr 23 10:20:32 2019

@author: Admin
"""
import random
import math

import tensorflow as tf

import numpy as np
import xlwt
import xlrd
import ctypes
import time

import matplotlib.pyplot as plt

from ctypes import *
from ctypes import c_double


lib = ctypes.cdll.LoadLibrary("F_Dll.dll")



MAX_EPISODES = 1
MAX_EP_STEPS = 1
LR_A = 0.001    # learning rate for actor 原值为0.001
LR_C = 0.002    # learning rate for critic 原值为0.002
GAMMA = 0.9     # reward discount  原值为0.9
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000   # 原值为100000
BATCH_SIZE = 32 # 原值为32
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'

class DDPG(object):

    def __init__(self, a_dim, s_dim, a_bound,):

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)

        self.pointer = 0

        self.sess = tf.Session()



        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')

        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')

        self.R = tf.placeholder(tf.float32, [None, 1], 'r')



        self.a = self._build_a(self.S,)

        q = self._build_c(self.S, self.a, )

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')

        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement



        def ema_getter(getter, name, *args, **kwargs):

            return ema.average(getter(name, *args, **kwargs))



        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation

        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters

        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)



        a_loss = - tf.reduce_mean(q)  # maximize the q

        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)



        with tf.control_dependencies(target_update):    # soft replacement happened at here

            q_target = self.R + GAMMA * q_

            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)

            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)



        self.sess.run(tf.global_variables_initializer())



    def choose_action(self, s):

        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]



    def learn(self):

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)

        bt = self.memory[indices, :]

        bs = bt[:, :self.s_dim]

        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]

        br = bt[:, -self.s_dim - 1: -self.s_dim]

        bs_ = bt[:, -self.s_dim:]



        self.sess.run(self.atrain, {self.S: bs})

        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})



    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, a, [r], s_))

        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory

        self.memory[index, :] = transition

        self.pointer += 1



    def _build_a(self, s, reuse=None, custom_getter=None):

        trainable = True if reuse is None else False

        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):

            net = tf.layers.dense(s, 50, activation=tf.nn.relu, name='l1', trainable=trainable)

            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)

            return tf.multiply(a, self.a_bound, name='scaled_a')



    def _build_c(self, s, a, reuse=None, custom_getter=None):

        trainable = True if reuse is None else False

        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):

            n_l1 = 50

            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)

            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)

            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

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


NH = 44000
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

wf1 = c_double(0.0589)
wf2 = c_double(0.071)
wf3 = c_double(0.0835)

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


s_dim = 1  # 状态个数
a_dim = 1

a_bound = np.array([1.0],float)

ddpg = DDPG(a_dim, s_dim, a_bound)


var = 3  # control exploration  var = 3

a = 0
action_1 = 0

observation_S = np.array([0.0],float)
observation_S1 = np.array([0.0],float)
observation_u = np.array([0.0],float)
observation_a = np.array([0.0],float)

t1 = time.time()
wf = 0.02
NH_z = 49000
e_k_1 = 0
e_a = 0

for i in range(MAX_EPISODES):    

    
    s = np.array([0],float)
    
    ep_reward = 0    

    for j in range(MAX_EP_STEPS):
        

        
        e_k = float(NH_z) - float(NH)
        

        s = np.array([wf],float)   
        
        a = ddpg.choose_action(s)

        action = np.clip(np.random.normal(a, var), -1, 1)    # add randomness to action selection for exploration

        ZJBL =  float(action) * 0.03 + 0.05
        
        if ZJBL > 0.08:
            ZJBL = 0.08
        if ZJBL < 0.02:
            ZJBL = 0.02       
     
        wf = ZJBL 
        
        lib.Set_Engine_Input(c_double(wf),H,Ma)
        
        for k in range(300):
            lib.Engine_Calculate()
            lib.Updata_Engine_Time()
            
        NH = lib.Get_Engine_NH()
        NL = lib.Get_Engine_NL()
        NLc = lib.Get_Engine_NLc()
        NHc = lib.Get_Engine_NHc()    
        wa = lib.Get_Engine_Wat()
        Fn = lib.Get_Engine_Fn()
        Tt4 = lib.Get_Engine_Tt4()    
        Pt4 = lib.Get_Engine_Pt4()
        wg4 = lib.Get_Engine_wg4()
        Tt3 = lib.Get_Engine_Tt3()
        Pt3 = lib.Get_Engine_Pt3()
        wa3 = lib.Get_Engine_wa3()
        SMC = lib.Get_Engine_SMC()
        Tt2 = lib.Get_Engine_Tt2()        
        Pt2 = lib.Get_Engine_Pt2()
        wa2 = lib.Get_Engine_wa2()
        SMF = lib.Get_Engine_SMF()
        Tt5 = lib.Get_Engine_Tt5()
        Pt5 = lib.Get_Engine_Pt5()        
        wg5 = lib.Get_Engine_wg5()
        Tt6 = lib.Get_Engine_Tt6()
        Pt6 = lib.Get_Engine_Pt6()
        wg6 = lib.Get_Engine_wg6()      
     

        e_k = NH_z - NH       
        
        r = - abs(e_k)*0.1
        
        s_ = np.array([wf],float)
        
        ddpg.store_transition(s, action, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:

            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        
        observation_S = np.append(observation_S,NH)
        observation_S1 = np.append(observation_S1,NH_z)
        observation_u = np.append(observation_u,wf)
        observation_a = np.append(observation_a,action) 
        
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(r), 'Explore: %.2f' % var,'wf: %.2f'% wf)
            break    
        
        cont = cont + 1  

