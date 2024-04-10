import numpy as np
import matplotlib.pyplot as plt
from sys import path
import os
from multiprocessing import Process
from pandas import read_csv, DataFrame
from numba import jit
from datetime import datetime

class Kinematics:
    def __init__(self):
        self.sys_path = path[0]
        self.resource_path = os.path.join("..",self.sys_path,"Resources")
        self.temp_folder = os.path.join("..",self.sys_path,"temp")
        self.mexcess = read_csv(os.path.join(self.resource_path,'mexcess.csv')).to_numpy()
        
        self.stop: bool = True

        self.xdim: int = None
        self.ydim: int = None
        self.dead: int = None
        self.threshd: int = None
        self.zp: float = None
        self.mp: float = None
        self.ke: float  = None # kinetic energy in MeV/u
        self.ep: float = None # kinetic energy in MeV
        self.zt: float = None
        self.mt: float = None
        self.zr: float = None
        self.mr: float = None
        self.ze: float = None # Conservation of Z
        self.me: float = None # Conservation of A
        self.simple_cm: float = None # in rad
        self.nreactions: int = None
        self.ex: np.ndarray = None
        self.DETECTED1: np.ndarray = None
        self.DETECTED2: np.ndarray = None
        self.DETECTED3: np.ndarray = None
        self.vz1: np.ndarray = None
        self.vz3: np.ndarray = None
        self.vz3: np.ndarray = None

    def setKinematics(self) -> None:
        '''
        Sets Variables for simulations
        '''
        # The Q value is the difference between the incoming and outgoing masses, expressed in MeV
        amu = 931.4941024

        try:
            mexp = self.mexcess[self.zp, self.mp-self.zp]
            mext = self.mexcess[self.zt, self.mt-self.zt]
            mexr = self.mexcess[self.zr, self.mr-self.zr]
            mexe = self.mexcess[self.ze, self.me-self.ze]

            if sum(np.isnan(np.array([mexp,mext,mexr,mexe]))) > 0:
                raise Exception
        except:
            print("Missing mass excess for selected nuclei, proceeding without exact numbers")
            mexp=0
            mext=0
            mexr=0
            mexe=0
        
        self.mp = (float(self.mp)*float(amu) + float(mexp)) / float(amu) # convert mass to amu units
        self.mt = (self.mt*amu + mext) / amu # convert mass to amu units
        self.mr = (self.mr*amu + mexr) / amu # convert mass to amu units
        self.me = (self.me*amu + mexe) / amu # convert mass to amu units
        self.Q = (self.mp + self.mt - self.mr - self.me) * amu - self.ex # in MeV

        print("Q:",self.Q)
        self.stop = False

        return

    def determineDetected(self) -> None:
        if self.stop:
            return 
        
        now = f"{datetime.now():%Y_%m_%d-%H_%M}"

        self.simpleLoop1()
        if len(self.vz1) <= 0:
            raise Exception
        
        export_df = DataFrame({"vz":self.vz1,
                               "cm":np.ones(len(self.vz1))*self.simple_cm,
                               "ex":np.zeros_like(self.vz1),
                               "detected":self.DETECTED1.astype(int)})
        export_df.to_csv(os.path.join(self.temp_folder,f"{now}-Simple_Sim_randomvz.csv"),index=False)
 
        self.simpleLoop2()
        if len(self.vz2) <= 0:
            raise Exception

        export_df = DataFrame({"vz":self.vz2,
                               "cm":self.Cm2,
                               "ex":np.zeros_like(self.vz2),
                               "detected":self.DETECTED2.astype(int)})
        export_df.to_csv(os.path.join(self.temp_folder,f"{now}-Simple_Sim_random_CMvz.csv"),index=False)

        p = Process(target=self.createFig)
        p.start()

    @jit(forceobj=True,looplift=True)
    def simpleLoop1(self):
        self.cm = np.ones(shape=self.nreactions) * self.simple_cm
        self.vz1 = np.random.uniform(0,self.xdim,size=self.nreactions)
        A1 = self.labAngle()
        A2 = self.labAngle2()
        Er = self.labEnergy(self.mr,self.me,A1)/self.mr
        Ee = self.labEnergy(self.me,self.mr,A2)/self.me

        NOTNA = ~np.logical_or(np.isnan(Er),np.isnan(Ee))
        self.cm = self.cm[NOTNA]
        self.vz1 = self.vz1[NOTNA]
        A1 = A1[NOTNA]
        A2 = A2[NOTNA]
        # Er = Er[NOTNA]
        # Ee = Ee[NOTNA]

        LT = A1 < np.pi/2
        GT = ~LT
        y1 = np.zeros_like(A1)
        y1[LT] = np.tan(A1[LT]) * (self.xdim-self.vz1[LT])
        y1[GT] = np.tan(A1[GT]) * self.vz1[GT]

        LT = A2 < np.pi/2
        GT = ~LT
        y2 = np.zeros_like(A2)
        y2[LT] = np.tan(A2[LT]) * (self.xdim-self.vz1[LT])
        y2[GT] = np.tan(A2[GT]) * self.vz1[GT]

        self.DETECTED1 = np.logical_and(y1 >= self.dead,y2 <= self.threshd)
        self.vz1 = self.vz1
        return
    
    @jit(forceobj=True,looplift=True)
    def simpleLoop2(self):
        self.cm = np.random.uniform(0,np.pi,size=self.nreactions)
        self.vz2 = np.random.uniform(0,self.xdim,size=self.nreactions)
        A1 = self.labAngle()
        A2 = self.labAngle2()
        Er = self.labEnergy(self.mr,self.me,A1)/self.mr
        Ee = self.labEnergy(self.me,self.mr,A2)/self.me

        NOTNA = ~np.logical_or(np.isnan(Er),np.isnan(Ee))
        self.cm = self.cm[NOTNA]
        self.vz2 = self.vz2[NOTNA]
        A1 = A1[NOTNA]
        A2 = A2[NOTNA]

        LT = A1 < np.pi/2
        GT = ~LT
        y1 = np.zeros_like(A1)
        y1[LT] = np.tan(A1[LT]) * (self.xdim-self.vz2[LT])
        y1[GT] = np.tan(A1[GT]) * self.vz2[GT]

        LT = A2 < np.pi/2
        GT = ~LT
        y2 = np.zeros_like(A2)
        y2[LT] = np.tan(A2[LT]) * (self.xdim-self.vz2[LT])
        y2[GT] = np.tan(A2[GT]) * self.vz2[GT]

        self.DETECTED2 = np.logical_and(y1 >= self.dead,y2 <= self.threshd)
        self.vz2 = self.vz2
        self.Cm2 = self.cm

        return

    def determineEnergy(self) -> None:
        if self.stop:
            return
        
        now = f"{datetime.now():%Y_%m_%d-%H_%M}"
        
        self.ENloop1()
        export_df = DataFrame({"vz":np.ones_like(self.Energy11)*self.xdim/2,
                               "cm":self.Cm1,
                               "er":self.Energy11,
                               "ee":self.Energy12,
                               "ex":np.zeros_like(self.Energy11),
                               "detected":self.DETECTED1.astype(int)})
        export_df.to_csv(os.path.join(self.temp_folder,f"{now}-EN_fixedvz_0ex.csv"),index=False)

        self.ENloop2()
        export_df = DataFrame({"vz":self.vz2,
                               "cm":self.Cm2,
                               "er":self.Energy21,
                               "ee":self.Energy22,
                               "ex":np.zeros_like(self.vz2),
                               "detected":self.DETECTED2.astype(int)})
        export_df.to_csv(os.path.join(self.temp_folder,f"{now}-EN_0ex.csv"),index=False)

        self.ENloop3()
        export_df = DataFrame({"vz":self.vz3,
                               "cm":self.Cm3,
                               "er":self.Energy31,
                               "ee":self.Energy32,
                               "ex":self.Excite3,
                               "detected":self.DETECTED3.astype(int)})
        export_df.to_csv(os.path.join(self.temp_folder,f"{now}-EN_randomex.csv"),index=False)

        p = Process(target=self.createENFig)
        p.start()

    @jit(forceobj=True,looplift=True)
    def ENloop1(self) -> None:
        vz = self.xdim / 2
        self.cm = np.random.uniform(0,np.pi,size=self.nreactions)
        A1 = self.labAngle()
        A2 = self.labAngle2()
        Er = self.labEnergy(self.mr,self.me,A1)/self.mr
        Ee = self.labEnergy(self.me,self.mr,A2)/self.me

        NOTNA = ~np.logical_or(np.isnan(Er),np.isnan(Ee))
        self.cm = self.cm[NOTNA]
        A1 = A1[NOTNA]
        A2 = A2[NOTNA]
        Er = Er[NOTNA]
        Ee = Ee[NOTNA]

        LT = A1 < np.pi/2
        GT = ~LT
        y1 = np.zeros_like(A1)
        y1[LT] = np.tan(A1[LT]) * (self.xdim-vz)
        y1[GT] = np.tan(A1[GT]) * vz

        LT = A2 < np.pi/2
        GT = ~LT
        y2 = np.zeros_like(A2)
        y2[LT] = np.tan(A2[LT]) * (self.xdim-vz)
        y2[GT] = np.tan(A2[GT]) * vz

        self.DETECTED1 = np.logical_and(y1 >= self.dead,y2 <= self.threshd)
        self.Cm1 = self.cm
        self.Energy11 = Er
        self.Energy12 = Ee

        return
    
    @jit(forceobj=True,looplift=True)
    def ENloop2(self) -> None:
        self.cm = np.random.uniform(0,np.pi,size=self.nreactions)
        self.vz2 = np.random.uniform(0,self.xdim,size=self.nreactions)
        A1 = self.labAngle()
        A2 = self.labAngle2()
        Er = self.labEnergy(self.mr,self.me,A1)/self.mr
        Ee = self.labEnergy(self.me,self.mr,A2)/self.me

        NOTNA = ~np.logical_or(np.isnan(Er),np.isnan(Ee))
        self.cm = self.cm[NOTNA]
        self.vz2 = self.vz2[NOTNA]
        A1 = A1[NOTNA]
        A2 = A2[NOTNA]
        Er = Er[NOTNA]
        Ee = Ee[NOTNA]

        LT = A1 < np.pi/2
        GT = ~LT
        y1 = np.zeros_like(A1)
        y1[LT] = np.tan(A1[LT]) * (self.xdim-self.vz2[LT])
        y1[GT] = np.tan(A1[GT]) * self.vz2[GT]

        LT = A2 < np.pi/2
        GT = ~LT
        y2 = np.zeros_like(A2)
        y2[LT] = np.tan(A2[LT]) * (self.xdim-self.vz2[LT])
        y2[GT] = np.tan(A2[GT]) * self.vz2[GT]

        self.DETECTED2 = np.logical_and(y1 >= self.dead,y2 <= self.threshd)
        self.vz2 = self.vz2
        self.Cm2 = self.cm
        self.Energy21 = Er
        self.Energy22 = Ee

        return

    @jit(forceobj=True,looplift=True)
    def ENloop3(self) -> None:
        amu = 931.4941024 # MeV/U

        self.cm = np.random.uniform(0,np.pi,size=self.nreactions)
        self.vz3 = np.random.uniform(0,self.xdim,size=self.nreactions)
        Ex = np.random.uniform(-20,20,size=self.nreactions)
        self.Q = (self.mp + self.mt - self.mr - self.me) * amu - Ex
        A1 = self.labAngle()
        A2 = self.labAngle2()
        Er = self.labEnergy(self.mr,self.me,A1)/self.mr
        Ee = self.labEnergy(self.me,self.mr,A2)/self.me

        NOTNA = ~np.logical_or(np.isnan(Er),np.isnan(Ee))
        self.cm = self.cm[NOTNA]
        self.vz3 = self.vz3[NOTNA]
        Ex = Ex[NOTNA]
        self.Q = self.Q[NOTNA]
        A1 = A1[NOTNA]
        A2 = A2[NOTNA]
        Er = Er[NOTNA]
        Ee = Ee[NOTNA]

        LT = A1 < np.pi/2
        GT = ~LT
        y1 = np.zeros_like(A1)
        y1[LT] = (self.xdim-self.vz3[LT]) * np.tan(A1[LT])
        y1[GT] = self.vz3[GT] * np.tan(A1[GT]) 

        LT = A2 < np.pi/2
        GT = ~LT
        y2 = np.zeros_like(A2)
        y2[LT] = (self.xdim-self.vz3[LT]) * np.tan(A2[LT])
        y2[GT] = self.vz3[GT] * np.tan(A2[GT])

        self.DETECTED3 = np.logical_and(y1 >= self.dead,y2 <= self.threshd)
        self.vz3 = self.vz3
        self.Cm3 = self.cm
        self.Energy31 = Er
        self.Energy32 = Ee
        self.Excite3 = Ex

        return

    def createFig(self) -> None:
        fig,ax = plt.subplots(nrows=3,ncols=1,dpi=150)

        ax[0].set_xlabel("Vertex of Reaction")
        ax[0].set_ylabel("Counts")
        ax[0].set_title(f"Number of detections for cm = {self.simple_cm*180/np.pi}")
        ax[0].set_facecolor('#ADD8E6')
        ax[0].set_axisbelow(True)
        ax[0].yaxis.grid(color='white', linestyle='-')
        ax[0].hist(self.vz1[self.DETECTED1],bins=100,range=(0,self.xdim))

        ax[1].set_xlabel("CM Angle (rad)")
        ax[1].set_ylabel("Counts")
        ax[1].set_title(f"Number of detections with random cm and vz")
        ax[1].set_facecolor('#ADD8E6')
        ax[1].set_axisbelow(True)
        ax[1].yaxis.grid(color='white', linestyle='-')
        ax[1].hist(self.Cm2[self.DETECTED2],bins=180,range=(0,np.pi))

        ax[2].set_xlabel("Vertex")
        ax[2].set_ylabel("Counts")
        ax[2].set_title(f"Number of detections with random cm and vz")
        ax[2].set_facecolor('#ADD8E6')
        ax[2].set_axisbelow(True)
        ax[2].yaxis.grid(color='white', linestyle='-')
        ax[2].hist(self.vz2[self.DETECTED2],bins=100,range=(0,self.xdim))

        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_folder,f"{datetime.now():%Y_%m_%d-%H_%M}_simple.png"),format="png")
        plt.cla()
        plt.clf()
        plt.close('all')

    def createENFig(self) -> None:
        size = 4*(1.05 - (1-np.exp(-2E-4 * self.nreactions)))
        fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(10,8))

        ax[0,0].set_xlabel("CM angle")
        ax[0,0].set_ylabel("Energy (MeV/U)")
        ax[0,0].set_title(f"vz = {self.xdim / 2}, random CM")
        ax[0,0].set_facecolor('#ADD8E6')
        ax[0,0].set_axisbelow(True)
        ax[0,0].yaxis.grid(color='white', linestyle='-')
        ax[0,0].scatter(x=self.Cm1[self.DETECTED1],y=self.Energy11[self.DETECTED1],s=size,c="#ff7f0e")
        ax[0,0].scatter(x=self.Cm1[self.DETECTED1],y=self.Energy12[self.DETECTED1],s=size,c="#1f77b4")
        ax[0,0].scatter(x=-100,y=-100,s=10,c="#ff7f0e",label="Er")
        ax[0,0].scatter(x=-100,y=-100,s=10,c="#1f77b4",label="Ee")
        ax[0,0].set_xlim(-0.1,max(self.Cm1)+0.1)
        ax[0,0].set_ylim(-2)
        ax[0,0].legend()

        ax[0,1].axis('off')
        ax[0,2].axis('off')

        ax[1,0].set_xlabel("CM angle")
        ax[1,0].set_ylabel("Energy (MeV/U)")
        ax[1,0].set_title(f"Random vz, CM")
        ax[1,0].set_facecolor('#ADD8E6')
        ax[1,0].set_axisbelow(True)
        ax[1,0].yaxis.grid(color='white', linestyle='-')
        ax[1,0].scatter(x=self.Cm2[self.DETECTED2],y=self.Energy21[self.DETECTED2],s=size,c="#ff7f0e")
        ax[1,0].scatter(x=self.Cm2[self.DETECTED2],y=self.Energy22[self.DETECTED2],s=size,c="#1f77b4")

        ax[1,1].set_xlabel("Vertex")
        ax[1,1].set_facecolor('#ADD8E6')
        ax[1,1].set_axisbelow(True)
        ax[1,1].yaxis.grid(color='white', linestyle='-')
        ax[1,1].scatter(x=self.vz2[self.DETECTED2],y=self.Energy21[self.DETECTED2],s=size,c="#FC776AFF")
        ax[1,1].scatter(x=self.vz2[self.DETECTED2],y=self.Energy22[self.DETECTED2],s=size,c="#5B84B1FF")
        ax[1,1].scatter(x=-100,y=-100,s=10,c="#FC776AFF",label="Er")
        ax[1,1].scatter(x=-100,y=-100,s=10,c="#5B84B1FF",label="Ee")
        ax[1,1].set_xlim(-2,max(self.vz2)+2)
        ax[1,1].set_ylim(-2)
        ax[1,1].legend()

        ax[1,2].axis('off')

        ax[2,0].set_xlabel("CM angle")
        ax[2,0].set_ylabel("Energy (MeV/U)")
        ax[2,0].set_title(f"Random vz, CM, ex")
        ax[2,0].set_facecolor('#ADD8E6')
        ax[2,0].set_axisbelow(True)
        ax[2,0].yaxis.grid(color='white', linestyle='-')
        ax[2,0].scatter(x=self.Cm3[self.DETECTED3],y=self.Energy31[self.DETECTED3],s=size,c="#ff7f0e")
        ax[2,0].scatter(x=self.Cm3[self.DETECTED3],y=self.Energy32[self.DETECTED3],s=size,c="#1f77b4")

        ax[2,1].set_xlabel("Vertex")
        ax[2,1].set_facecolor('#ADD8E6')
        ax[2,1].set_axisbelow(True)
        ax[2,1].yaxis.grid(color='white', linestyle='-')
        ax[2,1].scatter(x=self.vz3[self.DETECTED3],y=self.Energy31[self.DETECTED3],s=size,c="#FC776AFF")
        ax[2,1].scatter(x=self.vz3[self.DETECTED3],y=self.Energy32[self.DETECTED3],s=size,c="#5B84B1FF")

        ax[2,2].set_xlabel("Ex")
        ax[2,2].set_facecolor('#ADD8E6')
        ax[2,2].set_axisbelow(True)
        ax[2,2].yaxis.grid(color='white', linestyle='-')
        ax[2,2].scatter(x=self.Excite3[self.DETECTED3],y=self.Energy31[self.DETECTED3],s=size,c="#5F4B8BFF")
        ax[2,2].scatter(x=self.Excite3[self.DETECTED3],y=self.Energy32[self.DETECTED3],s=size,c="#E69A8DFF")
        ax[2,2].scatter(x=-100,y=-100,s=10,c="#5F4B8BFF",label="Er")
        ax[2,2].scatter(x=-100,y=-100,s=10,c="#E69A8DFF",label="Ee")
        ax[2,2].set_xlim(min(self.Excite3[self.DETECTED3])-0.2,max(self.Excite3[self.DETECTED3])+0.2)
        ax[2,2].set_ylim(-1)
        ax[2,2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_folder,f"{datetime.now():%Y_%m_%d-%H_%M}_energy.png"),format="png")
        plt.cla()
        plt.clf()
        plt.close('all')
    
    @jit(forceobj=True,looplift=True)
    def labAngle(self) -> np.ndarray:
        gam: np.ndarray = np.sqrt(self.mp*self.mr/self.mt/self.me*self.ep/(self.ep+self.Q*(1+self.mp/self.mt)))
        lab: np.ndarray = np.arctan2(np.sin(self.cm),gam-np.cos(self.cm))
        return lab
    
    @jit(forceobj=True,looplift=True)
    def labAngle2(self) -> np.ndarray:
        gam: np.ndarray = np.sqrt(self.mp*self.me/self.mt/self.mr*self.ep/(self.ep+self.Q*(1+self.mp/self.mt)))
        lab: np.ndarray = np.arctan2(np.sin(self.cm),gam+np.cos(self.cm))
        return lab

    @jit(forceobj=True,looplift=True)
    def labEnergy(self,mr,me,th) -> np.ndarray:
        delta: np.ndarray = np.sqrt(self.mp*mr*self.ep*np.cos(th)**2 + (me+mr)*(me*self.Q+(me-self.mp)*self.ep))
        fir: np.ndarray = np.sqrt(self.mp*mr*self.ep)*np.cos(th)
        e1: np.ndarray = (fir + delta) / (me+mr)
        e2: np.ndarray = (fir - delta) / (me+mr)
        e1 = e1**2
        e2 = e2**2
        gam: np.ndarray = np.sqrt(self.mp*mr/self.mt/me*self.ep/(self.ep+self.Q*(1+self.mp/self.mt)))
        arg: np.ndarray = np.sin(self.cm)/(gam-np.cos(self.cm))
        der: np.ndarray = 1/(1+arg**2)*(gam*np.cos(self.cm)-1)/(gam-np.cos(self.cm))**2

        m: np.ndarray = der < 0
        e2[m] = e1[m]

        return e2