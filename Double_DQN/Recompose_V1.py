
"""
Created on Tue Dec 18 20:47:44 2018

@author: Admin
"""
from exlread import OpenExl

data_gk = OpenExl('data.xlsx', 0)

from ctypes import *
from scipy.interpolate import interp1d   #插值程序


import os
import numpy as np
import xlrd

import xlwt

#file=xlwt.Workbook(encoding='utf-8',style_compression=0)
#sheet=file.add_sheet('aa')

class MyEngine:
    def __init__(self):
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
 #%% 新控制规律           
#        self.nl=([0,4876,5024,5595,6287,6769,7130,7476,7616,7761,7969,8116,8254,8504,8636,8778,8873,9234,9993,10200])
#        self.pi=([1,1.213,1.233,1.293,1.393,1.473,1.537,1.605,1.744,1.709,1.784,1.86,1.942,2.008,2.087,2.167,2.3,2.519,2.731,2.755])
##%% 旧的控制规律
#        self.nl=([1000,4400,5000,5500,5700,6000,6300,6600,6800,7000,7250,7500,7750,8000,8250,8500,8750,9000,9250,9500,9700,10000,10200])
#        self.pi=([1,1.2017,1.2789,1.3594,1.3884,1.4409,1.4955,1.566,1.6222,1.6744,1.7563,1.8431,1.9524,2.102,2.2686,2.4555,2.6421,2.8625,3.051,3.2462,3.3651,3.5042,3.5945])
#%% 高空控制规律
        self.nl=([0,7708,7888,8061,8261,8480,8688,9080,9265,9446,9639,9928,10200,15800])
        self.pi=([[1,1.891,2.008,2.172,2.295,2.413,2.525,2.735,2.845,2.967,3.089,3.215,3.314,3.848]])



# 推力闭环算法
        self.Kp_Fn = 0.5
        self.Ki_Fn = 0.5
        self.xk_Fn = 0
#        self.xk_Fn = 5000
        self.a_Fn = 1
        self.b_Fn = 0.125
        self.c_Fn = 0.08
        self.d_Fn = 0.005
        self.NL_in = 9550
        self.NL_out = 9550

        self.NLc=9600


#Nl闭环算法  频域校正后的矩阵
        self.A = np.matrix([[1,0],[0,0.9704]])
        self.B = np.matrix([[0.0004921],[-0.002137]])
        self.C = np.matrix([[0.0004781,-0.002137]])
        self.D = np.matrix([[2.436e-6]])

        self.xk=np.matrix([[0],[0]])


#风扇压比闭环算法，PID积分项离散化后abcd
        self.xk_pi=0
        self.a_pi=1
        self.b_pi=0.125
        self.c_pi=0.08
        self.d_pi=0.005

        self.Kp=-0.02
        self.Ki=-0.02
        self.A8_in=0.2 #A8初值
        self.A8_out=0.2

        self.pi_b=3.0 #压比反馈初值
        self.pic=3.0
        self.t=0
        self.t1 =0
        self.Fnc = 1300
        self.e0 = 0
        self.e1 = 0
        self.e2 = 0
        self.pic = 0
        self.bbb = []
        self.flag = 0
        
        
        self.m_Engine = c_longdouble()
        self.m_EngineInPut = EngineInPut();
        self.m_EngineOutPut = EngineOutPut()        
        
        
        self.m_EngineInPut.P0=12111.8
        self.m_EngineInPut.T0=244.3812
        self.m_EngineInPut.ConT0=244.3812
        self.m_EngineInPut.ConP0=18462.5013
        self.m_EngineInPut.ConP2=18462.5013
        self.m_EngineInPut.Time=0.01
        self.m_EngineInPut.Wf=0.3985
        self.m_EngineInPut.Angle_CompressVane=7.084529  #1.78994
        self.m_EngineInPut.AddPower=0.0
        self.m_EngineInPut.Wf_After=0
        self.m_EngineInPut.Angle_FanVane=0  #0.7569
        self.m_EngineInPut.Area_Spout=0.239223
        self.m_EngineInPut.Area_Spout_A9=0.37 #0.32086598
        self.m_EngineInPut.Angle_CoreFanVane=-5
        self.m_EngineInPut.Angle_Guide_LowTurbo=0
        self.m_EngineInPut.Area_OutCulvert_Core=0.060277
        self.m_EngineInPut.Area_OutCulvert=0
        self.m_EngineInPut.Excursion_Eff_Compress=1
        self.m_EngineInPut.Excursion_Eff_Fan=1
        self.m_EngineInPut.Excursion_Inertia_Fan=1
        self.m_EngineInPut.Excursion_Inertia_Compress=1
        self.m_EngineInPut.Scale_Gas_Elicit=0
        self.m_EngineInPut.On_Off_EffAdapt=0
        
        self.Engine = cdll.LoadLibrary('F_Dll.dll')
        self.Engine.CreateEngine.restype = EngineOutPut
        self.Engine.EngineStepGo.restype = EngineOutPut
        
        self.m_EngineOutPut = self.Engine.CreateEngine(byref(self.m_Engine))
   
    def stepgo(self):        
        
        self.m_EngineOutPut = self.Enine.EngineStepGo(self.m_Engine,self.m_EngineInPut)
    
    def DestroyEngine(self):
        self.Engine.DestroyEngine(self.m_Engine)
    
    def QD_start(self):        
 
        
        for i in range(data_gk.table.nrows):

            self.bbb=data_gk.table.row_values(i)
            self.m_EngineInPut.Angle_CompressVane = self.bbb[0]
            self.m_EngineInPut.Angle_CoreFanVane = self.bbb[1]
            self.m_EngineInPut.Angle_FanVane = self.bbb[2]
            self.m_EngineInPut.Area_OutCulvert = self.bbb[3]
            self.m_EngineInPut.Area_OutCulvert_Core = self.bbb[4]
            self.m_EngineInPut.Area_Spout = self.bbb[5]
            self.m_EngineInPut.Area_Spout_A9 = self.bbb[6]
            self.m_EngineInPut.AddPower = self.bbb[7]
            self.m_EngineInPut.Wf = self.bbb[8]            
        
            if (self.m_EngineOutPut.nCompress_C > 10200) or (self.flag == 1):
            
                self.e0 = float (self.Fnc - self.m_EngineOutPut.Thrust_Net)
                
                
                
                self.u_Fn = self.e0*self.Ki_Fn + float(self.NL_out-self.NL_in)*1
                self.y_Fn = self.c_Fn*self.xk_Fn + self.d_Fn*self.u_Fn
                self.xk_Fn = self.a_Fn*self.xk_Fn + self.b_Fn*self.u_Fn
                
                self.NL_in = self.e0*self.Kp_Fn + self.y_Fn
                
                if self.NL_in > 10200:
                    self.NL_out = 10200
                elif self.NL_in < 4400:
                    self.NL_out = 4400
                else:
                    self.NL_out = self.NL_in
              
                self.NLc = self.NL_out
                
                self.NLc = self.NLc + 5000
               
                #转速闭环控制算法
                self.e1 = float ( self.NLc - self.m_EngineOutPut.nFan_C )
                
                self.m_EngineInPut.Wf = float(np.add(np.dot(self.C,self.xk),np.dot(self.D,self.e1)))
               
                self.m_EngineInPut.Wf = self.m_EngineInPut.Wf + 0.37
                
                self.xk = np.add(np.dot(self.A,self.xk),np.dot(self.B,self.e1))
                
                #压比闭环控制算法
                self.f1=interp1d(self.nl,self.pi,kind='linear')
                self.pic=self.f1(self.m_EngineOutPut.nFan_C)    #插值计算压比指令        
                
#                self.pic=self.f1(self.m_EngineOutPut.nFan_C) - 0.035    # 更改20190211
                
                self.e2=float(self.pic-self.pi_b)
                
                self.u_pi = self.e2*self.Ki + float(self.A8_out-self.A8_in)*1
                self.y_pi = self.c_pi*self.xk_pi + self.d_pi*self.u_pi
                self.xk_pi = self.a_pi*self.xk_pi + self.b_pi*self.u_pi
                
                self.A8_in = self.e2*self.Kp + self.y_pi
                
                if self.A8_in > 0.6:
                    self.A8_out = 0.6
                elif self.A8_in < 0.2:
                    self.A8_out = 0.2
                else:
                    self.A8_out = self.A8_in
                
                self.m_EngineInPut.Area_Spout = self.A8_out              
                
#                self.m_EngineInPut.Area_Spout_A9 = self.A8_out
                
                self.flag = 1
                
#                sheet.write(i,0,self.m_EngineOutPut.Loc_Fan)
#                sheet.write(i,1,self.m_EngineOutPut.sFc_Net)
#                sheet.write(i,2,float(self.Fnc))
#                sheet.write(i,3,float(self.m_EngineOutPut.Thrust_Net))
#                sheet.write(i,4,float(self.NLc))
#                sheet.write(i,5,self.m_EngineOutPut.nFan_C)    
#                sheet.write(i,6,float(self.pic))
#                sheet.write(i,7,float(self.m_EngineOutPut.tP25/self.m_EngineOutPut.tP2))    
#                sheet.write(i,8,self.m_EngineInPut.Area_Spout)    
#                sheet.write(i,9,self.m_EngineInPut.Wf)
#                sheet.write(i,10,self.m_EngineOutPut.nCompress_C)
#                sheet.write(i,11,self.m_EngineInPut.AddPower)
                
#            self.t1 = self.t1 + self.m_EngineInPut.Time    
            self.m_EngineOutPut = self.Engine.EngineStepGo(self.m_Engine,self.m_EngineInPut)
            
            self.pi_b=float(self.m_EngineOutPut.tP25)/float(self.m_EngineOutPut.tP2)
            
#            file.save('test.xls')
    
    def XunHang(self,S_):
        
        self.e0 = float (self.Fnc - self.m_EngineOutPut.Thrust_Net)
        
        self.e0 = self.e0 * 10  # 推力闭环速度加快
        
        self.u_Fn = self.e0*self.Ki_Fn + float(self.NL_out-self.NL_in)*1
        self.y_Fn = self.c_Fn*self.xk_Fn + self.d_Fn*self.u_Fn
        self.xk_Fn = self.a_Fn*self.xk_Fn + self.b_Fn*self.u_Fn
        
        self.NL_in = self.e0*self.Kp_Fn + self.y_Fn
        
        if self.NL_in > 10200:
            self.NL_out = 10200
        elif self.NL_in < 4400:
            self.NL_out = 4400
        else:
            self.NL_out = self.NL_in
      
        self.NLc = self.NL_out
        
        self.NLc = self.NLc + 5000
#        self.NLc = self.NLc + 0
       
        #转速闭环控制算法
        self.e1 = float ( self.NLc - self.m_EngineOutPut.nFan_C )
        
        self.m_EngineInPut.Wf = float(np.add(np.dot(self.C,self.xk),np.dot(self.D,self.e1)))
        
        self.m_EngineInPut.Wf = self.m_EngineInPut.Wf + 0.37
        
        self.xk = np.add(np.dot(self.A,self.xk),np.dot(self.B,self.e1))
        
        #压比闭环控制算法
        self.f1=interp1d(self.nl,self.pi,kind='linear')
        self.pic=self.f1(self.m_EngineOutPut.nFan_C)    #插值计算压比指令  
        
        self.pic = self.pic + S_
#        self.pic = self.pic + 0

        self.e2=float(self.pic-self.pi_b)
        
        self.u_pi = self.e2*self.Ki + float(self.A8_out-self.A8_in)*1
        self.y_pi = self.c_pi*self.xk_pi + self.d_pi*self.u_pi
        self.xk_pi = self.a_pi*self.xk_pi + self.b_pi*self.u_pi
        
        self.A8_in = self.e2*self.Kp + self.y_pi
        
        if self.A8_in > 0.6:
            self.A8_out = 0.6
        elif self.A8_in < 0.2:
            self.A8_out = 0.2
        else:
            self.A8_out = self.A8_in
        
        self.m_EngineInPut.Area_Spout = self.A8_out
#        self.m_EngineInPut.Area_Spout_A9 = self.A8_out
        
        self.m_EngineOutPut = self.Engine.EngineStepGo(self.m_Engine,self.m_EngineInPut)
        self.pi_b=float(self.m_EngineOutPut.tP25)/float(self.m_EngineOutPut.tP2)
        

        

#if __name__ == '__main__':
#    MyEngine = MyEngine()
#    MyEngine.QD_start()
#    MyEngine.DestroyEngine()
                
        
        
          