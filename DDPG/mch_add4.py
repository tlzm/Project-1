from ctypes import *
from scipy.interpolate import interp1d   #插值程序

from exlread import OpenExl

data_gk = OpenExl('data.xlsx', 0)

import xlwt
import xlrd
import time
import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pda
#%% 优化参数

b=0  # 赋初值
# 固定推力
Fnc = float(1500) 
#%%
#压比指令由低转转速插值得出，插值表
#nl=([0,4400,5000,5500,5700,6000,6300,6600,6800,7000,7250,7500,7750,8000,8250,8500,8750,9000,9250,9500,9700,10000,10200,15000])
#pi=([1,1.2017,1.2789,1.3594,1.3884,1.4409,1.4955,1.566,1.6222,1.6744,1.7563,1.8431,1.9524,2.102,2.2686,2.4555,2.6421,2.8625,3.051,3.2462,3.3651,3.5042,3.5945,5.0])
#nl=([0,4876,5024,5595,6287,6769,7130,7476,7616,7761,7969,8116,8254,8504,8636,8778,8873,9234,9993,10200])
#pi=([1,1.213,1.233,1.293,1.393,1.473,1.537,1.605,1.744,1.709,1.784,1.86,1.942,2.008,2.087,2.167,2.3,2.519,2.731,2.755])
nl=([0,7708,7888,8061,8261,8480,8688,9080,9265,9446,9639,9928,10200,13800])
pi=([[1,1.891,2.008,2.172,2.295,2.413,2.525,2.735,2.845,2.967,3.089,3.215,3.314,3.648]])


# 推力闭环算法
Kp_Fn = 0.5
Ki_Fn = 0.5
xk_Fn = 0
a_Fn = 1
b_Fn = 0.125
c_Fn = 0.08
d_Fn = 0.005
NL_in = 9550
NL_out = 9550

NLc=9600


#Nl闭环算法  频域校正后的矩阵
A=np.matrix([[1,0],[0,0.9704]])
B=np.matrix([[0.0004921],[-0.002137]])
C=np.matrix([[0.0004781,-0.002137]])
D=np.matrix([[2.436e-6]])
xk=np.matrix([[0],[0]])



#风扇压比闭环算法，PID积分项离散化后abcd
xk_pi=0
a_pi=1
b_pi=0.125
c_pi=0.08
d_pi=0.005

Kp=-0.02
Ki=-0.02
A8_in=0.2 #A8初值
A8_out=0.2

pi_b=3.0 #压比反馈初值
pic=3.0

file=xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet=file.add_sheet('aa')

t0= time.clock()
t=0
t1=0
i=1
#j=1
flag = 0

with open('power.txt') as file_object:
    power=file_object.readlines()

class EngineInPut(Structure):
    _fields_ = [
            ("P0",c_double),
            ("T0",c_double),
            ("ConT0",c_double),
            ("ConP0",c_double),
            ("ConP2",c_double),
            ("Time",c_double),
            ("Wf",c_double),
            ("Angle_CompressVane",c_double),
            ("AddPower",c_double),
            ("Wf_After",c_double),
            ("Angle_FanVane",c_double),
            ("Area_Spout",c_double),
            ("Area_Spout_A9",c_double),
            ("Angle_CoreFanVane",c_double),
            ("Angle_Guide_LowTurbo",c_double),
            ("Area_OutCulvert_Core",c_double),
            ("Area_OutCulvert",c_double)	,
            ("Excursion_Eff_Compress",c_double),
            ("Excursion_Eff_Fan",c_double),		
            ("Excursion_Inertia_Fan",c_double),
            ("Excursion_Inertia_Compress",c_double),	
            ("Scale_Gas_Elicit",c_double),
            ("On_Off_EffAdapt",c_int)
          ]
    
class EngineOutPut(Structure):
    _fields_ = [   
        ("nFan",c_double),
        ("nCompress",c_double),
        ("nFan_R",c_double),
        ("nCompress_R",c_double),
        ("nFan_C",c_double),
        ("nCoreFan_C",c_double),
        ("nCompress_C",c_double),
        ("nCompress_Cor",c_double),
        ("nCoreFan_Cor",c_double),
        ("nFan_Cor",c_double),
        ("nFan_COR_T2",c_double),
        ("nCoreFan_COR_T2",c_double),
        ("nCompress_COR_T2",c_double),
        ("Ma_Fan",c_double),
        ("Ma_CoreFan",c_double),
        ("Ma_HighCompress",c_double),
        ("B_Ma_Wf_Burn",c_double),
        ("Eff_Fire",c_double),
        ("tT2",c_double),
        ("tP2",c_double),
        ("tT25",c_double),
        ("tP25",c_double),
        ("tT28",c_double),
        ("tP28",c_double),
        ("tP31",c_double),
        ("sP31",c_double),
        ("tT6",c_double),        
        ("tP6",c_double),
        ("Loc_Compress",c_double),
        ("Loc_Fan",c_double),
        ("Loc_Fan_Core",c_double),
        ("Eff_Compress",c_double),
        ("Eff_Fan",c_double),
        ("Eff_CoreFan",c_double),
        ("Eff_OutCulvert",c_double),
        ("Eff_OutCulvert_Core",c_double),
        ("Eff_TurboHigh",c_double),
        ("Eff_TurboLow",c_double),
        ("Power_Dat_Fan",c_double),
        ("Power_Dat_Compress",c_double),
        ("Power_Compress_Else",c_double),
        ("Wf_Main",c_double),
        ("Eff_Adapt",c_double),
        ("tP6_Outer",c_double),
        ("tP6_CoreOuter",c_double),
        ("tP6_Inner",c_double),
        ("EPR",c_double),
        ("M_Engine_A9",c_double),
        ("V_Engine_A9",c_double),
        ("ConT9_Out_Engine",c_double),
        ("Thrust_Net",c_double),
        ("sFc_Net",c_double),
        ("Area_A9_All_Expand",c_double),
        ("HealthState",c_int),
        ("Iteration",c_int),
        ("Out_EngineInPut",EngineInPut)
        ]

#%%  定义变量和输入输出
m_Engine = c_longdouble()
m_EngineInPut = EngineInPut();
m_EngineOutPut = EngineOutPut()

m_EngineInPut.P0=12111.8
m_EngineInPut.T0=244.3812
m_EngineInPut.ConT0=244.3812
m_EngineInPut.ConP0=18462.5013
m_EngineInPut.ConP2=18462.5013
m_EngineInPut.Time=0.01
m_EngineInPut.Wf=0.3985
m_EngineInPut.Angle_CompressVane=1.78994
m_EngineInPut.AddPower=0.0
m_EngineInPut.Wf_After=0
m_EngineInPut.Angle_FanVane=0.7569
m_EngineInPut.Area_Spout=0.239223
m_EngineInPut.Area_Spout_A9=0.32086598
m_EngineInPut.Angle_CoreFanVane=-5
m_EngineInPut.Angle_Guide_LowTurbo=0
m_EngineInPut.Area_OutCulvert_Core=0.060277
m_EngineInPut.Area_OutCulvert=0
m_EngineInPut.Excursion_Eff_Compress=1
m_EngineInPut.Excursion_Eff_Fan=1
m_EngineInPut.Excursion_Inertia_Fan=1
m_EngineInPut.Excursion_Inertia_Compress=1
m_EngineInPut.Scale_Gas_Elicit=0
m_EngineInPut.On_Off_EffAdapt=0
#%% 调用dll
VCE = cdll.LoadLibrary('F_Dll.dll')
VCE.CreateEngine.restype = EngineOutPut
VCE.EngineStepGo.restype = EngineOutPut

m_EngineOutPut = VCE.CreateEngine(byref(m_Engine))
#%% 起动段
for i in range(data_gk.table.nrows):
    
    bbb=data_gk.table.row_values(i)
    
    m_EngineInPut.Angle_CompressVane = bbb[0]
    m_EngineInPut.Angle_CoreFanVane = bbb[1]
    m_EngineInPut.Angle_FanVane = bbb[2]
    m_EngineInPut.Area_OutCulvert = bbb[3]
    m_EngineInPut.Area_OutCulvert_Core = bbb[4]
    m_EngineInPut.Area_Spout = bbb[5]
    m_EngineInPut.Area_Spout_A9 = bbb[6]
    m_EngineInPut.AddPower = bbb[7]
    m_EngineInPut.Wf = bbb[8]     

    if (m_EngineOutPut.nCompress_C > 10200) or (flag == 1):
       
        e0 = float (Fnc - m_EngineOutPut.Thrust_Net)
        
        u_Fn = e0*Ki_Fn + float(NL_out-NL_in)*1
        y_Fn = c_Fn*xk_Fn + d_Fn*u_Fn
        xk_Fn = a_Fn*xk_Fn + b_Fn*u_Fn
        
        NL_in = e0*Kp_Fn + y_Fn
        
        if NL_in > 10200:
            NL_out = 10200
        elif NL_in < 4400:
            NL_out = 4400
        else:
            NL_out = NL_in
      
        NLc = NL_out
        
        NLc = NLc + 5000
       
        #转速闭环控制算法
        e1 = float ( NLc - m_EngineOutPut.nFan_C )
        
        m_EngineInPut.Wf = float(np.add(np.dot(C,xk),np.dot(D,e1)))
        m_EngineInPut.Wf = m_EngineInPut.Wf + 0.37
        xk = np.add(np.dot(A,xk),np.dot(B,e1))
        
        #压比闭环控制算法
        f1=interp1d(nl,pi,kind='linear')
        pic=f1(m_EngineOutPut.nFan_C)    #插值计算压比指令        

        e2=float(pic-pi_b)
        
        u_pi = e2*Ki + float(A8_out-A8_in)*1
        y_pi = c_pi*xk_pi + d_pi*u_pi
        xk_pi = a_pi*xk_pi + b_pi*u_pi
        
        A8_in = e2*Kp + y_pi
        
        if A8_in > 0.6:
            A8_out = 0.6
        elif A8_in < 0.2:
            A8_out = 0.2
        else:
            A8_out = A8_in
            
        m_EngineInPut.Area_Spout = A8_out
        m_EngineInPut.Area_Spout_A9 = A8_out  # A8 = A9
        
        flag = 1            
  
    m_EngineOutPut = VCE.EngineStepGo(m_Engine,m_EngineInPut)
    
    pi_b=float(m_EngineOutPut.tP25)/float(m_EngineOutPut.tP2)
    
#    sheet.write(i,0,m_EngineOutPut.Loc_Fan)
#    sheet.write(i,1,m_EngineOutPut.sFc_Net)
#    sheet.write(i,2,float(Fnc))
#    sheet.write(i,3,float(m_EngineOutPut.Thrust_Net))
#    sheet.write(i,4,float(NLc))
#    sheet.write(i,5,m_EngineOutPut.nFan_C)    
#    sheet.write(i,6,float(pic))
#    sheet.write(i,7,float(m_EngineOutPut.tP25/m_EngineOutPut.tP2))    
#    sheet.write(i,8,m_EngineInPut.Area_Spout)    
#    sheet.write(i,9,m_EngineInPut.Wf)
#    sheet.write(i,10,m_EngineOutPut.nCompress_C)
#    sheet.write(i,11,m_EngineInPut.AddPower) 
#    sheet.write(i,12,xk_Fn)

    
#    起动段结束
#%%
#j = 1
for j in range(60000):   
    
    e0 = float (Fnc - m_EngineOutPut.Thrust_Net)
    
    u_Fn = e0*Ki_Fn + float(NL_out-NL_in)*1
    y_Fn = c_Fn*xk_Fn + d_Fn*u_Fn
    xk_Fn = a_Fn*xk_Fn + b_Fn*u_Fn
    
    NL_in = e0*Kp_Fn + y_Fn
    
    if NL_in > 10200:
        NL_out = 10200
    elif NL_in < 4400:
        NL_out = 4400
    else:
        NL_out = NL_in
  
    NLc = NL_out
    
    NLc = NLc + 5000
   
    #转速闭环控制算法
    e1 = float ( NLc - m_EngineOutPut.nFan_C )
    
    m_EngineInPut.Wf = float(np.add(np.dot(C,xk),np.dot(D,e1)))
    m_EngineInPut.Wf = m_EngineInPut.Wf + 0.37
    xk = np.add(np.dot(A,xk),np.dot(B,e1))
    
    #压比闭环控制算法
    f1=interp1d(nl,pi,kind='linear')
    pic=f1(m_EngineOutPut.nFan_C)    #插值计算压比指令        

    e2=float(pic-pi_b)
    
    u_pi = e2*Ki + float(A8_out-A8_in)*1
    y_pi = c_pi*xk_pi + d_pi*u_pi
    xk_pi = a_pi*xk_pi + b_pi*u_pi
    
    A8_in = e2*Kp + y_pi
    
    if A8_in > 0.6:
        A8_out = 0.6
    elif A8_in < 0.2:
        A8_out = 0.2
    else:
        A8_out = A8_in
        
    m_EngineInPut.Area_Spout = A8_out
    m_EngineInPut.Area_Spout_A9 = A8_out  # A8 = A9
           
  
    m_EngineOutPut = VCE.EngineStepGo(m_Engine,m_EngineInPut)
    
  
    
    if j > 30000:
#        Fnc = float(4400)
#        m_EngineInPut.Angle_FanVane=10
        m_EngineInPut.Excursion_Eff_Fan=0.95
#        pi_b = pi_b * 1.1
#        m_EngineOutPut.tP25 = m_EngineOutPut.tP25 * 1.2
        
    pi_b=float(m_EngineOutPut.tP25)/float(m_EngineOutPut.tP2)
    
    
    sheet.write(j,0,m_EngineOutPut.Loc_Fan)
    sheet.write(j,1,m_EngineOutPut.sFc_Net)
    sheet.write(j,2,float(Fnc))
    sheet.write(j,3,float(m_EngineOutPut.Thrust_Net))
    sheet.write(j,4,float(NLc))
    sheet.write(j,5,m_EngineOutPut.nFan_C)    
    sheet.write(j,6,float(pic))
    sheet.write(j,7,float(m_EngineOutPut.tP25/m_EngineOutPut.tP2))    
    sheet.write(j,8,m_EngineInPut.Area_Spout)    
    sheet.write(j,9,m_EngineInPut.Wf)
    sheet.write(j,10,m_EngineOutPut.nCompress)
    sheet.write(j,11,m_EngineInPut.AddPower)

    
#%%  
VCE.DestroyEngine(m_Engine)

#sheet.write(0,0,'SMC')#%%第一列为时间
#sheet.write(0,1,'Sfc')
#sheet.write(0,2,'Fnc')
#sheet.write(0,3,'Fn')
#sheet.write(0,4,'NLc')
#sheet.write(0,5,'NL')
#sheet.write(0,6,'Pic')
#sheet.write(0,7,'Pi')
#sheet.write(0,8,'A8')
#sheet.write(0,9,'Wf')
#sheet.write(0,10,'NH')
#sheet.write(0,11,'tT6')
#sheet.write(0,12,'Fan_SMC')
#%%数据存储
file.save('mch_add4.xls')
#%% 画图功能
#huatu = 'mch_add4.xls'
#data = pda.read_excel(huatu)
#plt.plot(data.iloc[:,0],data.iloc[:,2],'r',data.iloc[:,0],data.iloc[:,3],'g')
#plt.show()

#%%cpu运行时间记录
t1=time.clock()
print("CPU run time is %f" % (t1-t0))