import numpy as np
import matplotlib.pyplot as plt
from sys import path
import os
from multiprocessing import Process
from pandas import read_csv, DataFrame

class Kinematics:
    def __init__(self):
        self.sys_path = path[0]
        self.resource_path = os.path.join("..",self.sys_path,"Resources")
        self.temp_folder = os.path.join("..",self.sys_path,"temp")
        self.mexcess = read_csv(os.path.join(self.resource_path,'mexcess.csv')).to_numpy()
        
        self.stop = True
        self.looping = False

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
        except:
            # GUI.errMessage("Missing Mexcess","Missing mass excess for selected nuclei, proceeding without exact numbers")
            mexp=0
            mext=0
            mexr=0
            mexe=0
        
        self.mp = (self.mp*amu + mexp) / amu # convert mass to amu units
        self.mt = (self.mt*amu + mext) / amu # convert mass to amu units
        self.mr = (self.mr*amu + mexr) / amu # convert mass to amu units
        self.me = (self.me*amu + mexe) / amu # convert mass to amu units
        self.Q = (self.mp + self.mt - self.mr - self.me) * amu - self.ex # in MeV

        try:
            self.labA1 = self.labAngle()
            self.labE1 = self.labEnergy(self.mr,self.me,self.labA1)/self.mr
            self.labA2 = self.labAngle2()
            self.labE2 = self.labEnergy(self.me,self.mr,self.labA2)/self.me # Swap mr and me for the beam-like
            print("A1:",self.labA1*180/np.pi,"E1:",self.labE1,"A2:",self.labA2*180/np.pi,"E2:",self.labE2,"Q:",self.Q)
            self.stop = False
        except:
            # GUI.toggleRunButtons("off")
            self.stop = True
            raise Exception

        return

    def determineDetected(self) -> None:
        if self.stop:
            return
        
        self.vz1 = []
        for _ in range(self.nreactions):
            vz = np.random.uniform(0.01,self.xdim-0.01)
            if self.labA1 < np.pi/2:
                y1 = (self.xdim-vz)*np.tan(self.labA1)
            else:
                y1 = vz*np.tan(self.labA1)
            if self.labA2 < np.pi/2:
                y2 = (self.xdim-vz)*np.tan(self.labA2)
            else:
                y2 = vz*np.tan(self.labA2)
            # The conditions for a successful event should be:
            # target-like Y > Deadzone (which means coming out of the dead zone)
            # AND beam-like Y < Threshold (which means entering in the zero degree detector)
            if y1 >= self.dead and y2 <= self.threshd:
                self.vz1.append(vz)

        if len(self.vz1) <= 0:
            raise Exception
            # GUI.errMessage("Invalid Reaction","Something went wrong, check reaction info")
            return
        
        export_df = DataFrame({"vz":self.vz1,
                               "cm":np.ones(len(self.vz1))*self.cm,
                               "ex":np.zeros_like(self.vz1)})
        export_df.to_csv(os.path.join(self.temp_folder,"Simple_Sim_randomvz.csv"),index=False)
        
        self.Cm2 = []
        self.vz2 = []
        for _ in range(self.nreactions):
            cm = np.random.uniform(0,np.pi)
            vz = np.random.uniform(0,self.xdim)
            A1 = self.labAngle()
            A2 = self.labAngle2()
            if A1 < np.pi/2:
                y1 = (self.xdim-vz)*np.tan(A1)
            else:
                y1 = vz*np.tan(A1)
            if A2 < np.pi/2:
                y2 = (self.xdim-vz)*np.tan(A2)
            else:
                y2 = vz*np.tan(A2)
            # The conditions for a successful event should be:
            # target-like Y > Deadzone (which means coming out of the dead zone)
            # AND beam-like Y < Threshold (which means entering in the zero degree detector)
            if y1 >= self.dead and y2 <= self.threshd:
                self.Cm2.append(cm)
                self.vz2.append(vz)

        if len(self.vz2) <= 0:
            raise Exception
            # GUI.errMessage(ValueError,"No particles detected")

        export_df = DataFrame({"vz":self.vz2,
                               "cm":self.Cm2,
                               "ex":np.zeros_like(self.vz2)})
        export_df.to_csv(os.path.join(self.temp_folder,"Simple_Sim_random_CMvz.csv"),index=False)

        # GUI.toggleRunButtons(state="on")

        p = Process(target=self.createFig)
        p.start()

    def determineEnergy(self) -> None:
        if self.stop:
            return
        
        self.looping = True
        amu = 931.4941024 # MeV/U
        
        vz = self.xdim / 2
        self.Energy11 = []
        self.Energy12 = []
        self.Cm1 = []

        for _ in range(self.nreactions):
            try:
                self.cm = np.random.uniform(0,np.pi)
                A1 = self.labAngle()
                A2 = self.labAngle2()
                Er = self.labEnergy(self.mr,self.me,A1)/self.mr
                Ee = self.labEnergy(self.me,self.mr,A2)/self.me
            except:
                continue
            if A1 < np.pi/2:
                y1 = (self.xdim-vz)*np.tan(A1)
            else:
                y1 = vz*np.tan(A1)
            if A2 < np.pi/2:
                y2 = (self.xdim-vz)*np.tan(A2)
            else:
                y2 = vz*np.tan(A2)
            if y1 >= self.dead and y2 <= self.threshd:
                self.Cm1.append(self.cm)
                self.Energy11.append(Er)
                self.Energy12.append(Ee)
        
        export_df = DataFrame({"vz":np.ones_like(self.Energy11)*vz,
                               "cm":self.Cm1,
                               "er":self.Energy11,
                               "ee":self.Energy12,
                               "ex":np.zeros_like(self.Energy11)})
        export_df.to_csv(os.path.join(self.temp_folder,"EN_fixedvz_0ex.csv"),index=False)

        self.vz2 = []
        self.Energy21 = []
        self.Energy22 = []
        self.Cm2 = []
        for _ in range(self.nreactions):
            try:
                self.cm = np.random.uniform(0,np.pi)
                vz = np.random.uniform(0,self.xdim)
                A1 = self.labAngle()
                A2 = self.labAngle2()
                Er = self.labEnergy(self.mr,self.me,A1)/self.mr
                Ee = self.labEnergy(self.me,self.mr,A2)/self.me
            except:
                continue
            if A1 < np.pi/2:
                y1 = (self.xdim-vz)*np.tan(A1)
            else:
                y1 = vz*np.tan(A1)
            if A2 < np.pi/2:
                y2 = (self.xdim-vz)*np.tan(A2)
            else:
                y2 = vz*np.tan(A2)
            if y1 >= self.dead and y2 <= self.threshd:
                self.vz2.append(vz)
                self.Cm2.append(self.cm)
                self.Energy21.append(Er)
                self.Energy22.append(Ee)

        export_df = DataFrame({"vz":self.vz2,
                               "cm":self.Cm2,
                               "er":self.Energy21,
                               "ee":self.Energy22,
                               "ex":np.zeros_like(self.vz2)})
        export_df.to_csv(os.path.join(self.temp_folder,"EN_0ex.csv"),index=False)

        self.vz3 = []
        self.Cm3 = []
        self.Energy31 = []
        self.Energy32 = []
        self.Excite3 = []
        for _ in range(self.nreactions):
            try:
                self.cm = np.random.uniform(0,np.pi)
                vz = np.random.uniform(0,self.xdim)
                Ex = np.random.uniform(0,9)
                self.Q = (self.mp + self.mt - self.mr - self.me) * amu - Ex
                A1 = self.labAngle()
                A2 = self.labAngle2()
                Er = self.labEnergy(self.mr,self.me,A1)/self.mr
                Ee = self.labEnergy(self.me,self.mr,A2)/self.me
            except:
                continue
            if A1 < np.pi/2:
                y1 = (self.xdim-vz)*np.tan(A1)
            else:
                y1 = vz*np.tan(A1)
            if A2 < np.pi/2:
                y2 = (self.xdim-vz)*np.tan(A2)
            else:
                y2 = vz*np.tan(A2)
            if y1 >= self.dead and y2 <= self.threshd:
                self.vz3.append(vz)
                self.Cm3.append(self.cm)
                self.Energy31.append(Er)
                self.Energy32.append(Ee)
                self.Excite3.append(Ex)

        export_df = DataFrame({"vz":self.vz3,
                               "cm":self.Cm3,
                               "er":self.Energy31,
                               "ee":self.Energy32,
                               "ex":self.Excite3})
        export_df.to_csv(os.path.join(self.temp_folder,"EN_randomex.csv"),index=False)

        self.looping = False
        # GUI.toggleRunButtons(state="on")

        p = Process(target=self.createENFig)
        p.start()

    def createFig(self) -> None:
        fig,ax = plt.subplots(nrows=3,ncols=1,dpi=150)
        ax[0].set_xlabel("Vertex of Reaction")
        ax[0].set_ylabel("Counts")
        ax[0].set_title(f"Number of detections for cm = {self.cm*180/np.pi}")
        ax[0].set_facecolor('#ADD8E6')
        ax[0].set_axisbelow(True)
        ax[0].yaxis.grid(color='white', linestyle='-')
        ax[0].hist(self.vz1,bins=100,range=(0,self.xdim))

        ax[1].set_xlabel("CM Angle (rad)")
        ax[1].set_ylabel("Counts")
        ax[1].set_title(f"Number of detections with random cm and vz")
        ax[1].set_facecolor('#ADD8E6')
        ax[1].set_axisbelow(True)
        ax[1].yaxis.grid(color='white', linestyle='-')
        ax[1].hist(self.Cm2,bins=180,range=(0,np.pi))

        ax[2].set_xlabel("Vertex")
        ax[2].set_ylabel("Counts")
        ax[2].set_title(f"Number of detections with random cm and vz")
        ax[2].set_facecolor('#ADD8E6')
        ax[2].set_axisbelow(True)
        ax[2].yaxis.grid(color='white', linestyle='-')
        ax[2].hist(self.vz1,bins=100,range=(0,self.xdim))

        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_folder,'fig1.jpg'),format="jpg")
        plt.show()
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
        ax[0,0].scatter(x=self.Cm1,y=self.Energy11,s=size,c="#ff7f0e")
        ax[0,0].scatter(x=self.Cm1,y=self.Energy12,s=size,c="#1f77b4")
        ax[0,0].scatter(x=-10,y=-10,s=10,c="#ff7f0e",label="Er")
        ax[0,0].scatter(x=-10,y=-10,s=10,c="#1f77b4",label="Ee")
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
        ax[1,0].scatter(x=self.Cm2,y=self.Energy21,s=size,c="#ff7f0e")
        ax[1,0].scatter(x=self.Cm2,y=self.Energy22,s=size,c="#1f77b4")

        ax[1,1].set_xlabel("Vertex")
        ax[1,1].set_facecolor('#ADD8E6')
        ax[1,1].set_axisbelow(True)
        ax[1,1].yaxis.grid(color='white', linestyle='-')
        ax[1,1].scatter(x=self.vz2,y=self.Energy21,s=size,c="#FC776AFF")
        ax[1,1].scatter(x=self.vz2,y=self.Energy22,s=size,c="#5B84B1FF")
        ax[1,1].scatter(x=-10,y=-10,s=10,c="#FC776AFF",label="Er")
        ax[1,1].scatter(x=-10,y=-10,s=10,c="#5B84B1FF",label="Ee")
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
        ax[2,0].scatter(x=self.Cm3,y=self.Energy31,s=size,c="#ff7f0e")
        ax[2,0].scatter(x=self.Cm3,y=self.Energy32,s=size,c="#1f77b4")

        ax[2,1].set_xlabel("Vertex")
        ax[2,1].set_facecolor('#ADD8E6')
        ax[2,1].set_axisbelow(True)
        ax[2,1].yaxis.grid(color='white', linestyle='-')
        ax[2,1].scatter(x=self.vz3,y=self.Energy31,s=size,c="#FC776AFF")
        ax[2,1].scatter(x=self.vz3,y=self.Energy32,s=size,c="#5B84B1FF")

        ax[2,2].set_xlabel("Ex")
        ax[2,2].set_facecolor('#ADD8E6')
        ax[2,2].set_axisbelow(True)
        ax[2,2].yaxis.grid(color='white', linestyle='-')
        ax[2,2].scatter(x=self.Excite3,y=self.Energy31,s=size,c="#5F4B8BFF")
        ax[2,2].scatter(x=self.Excite3,y=self.Energy32,s=size,c="#E69A8DFF")
        ax[2,2].scatter(x=-10,y=-10,s=10,c="#5F4B8BFF",label="Er")
        ax[2,2].scatter(x=-10,y=-10,s=10,c="#E69A8DFF",label="Ee")
        ax[2,2].set_xlim(-0.2,max(self.Excite3)+0.2)
        ax[2,2].set_ylim(-1)
        ax[2,2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_folder,'fig2.jpg'),format="jpg",dpi=300)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close('all')
        
    def labAngle(self) -> float:
        gam = np.sqrt(self.mp*self.mr/self.mt/self.me*self.ep/(self.ep+self.Q*(1+self.mp/self.mt)))
        lab = np.arctan2(np.sin(self.cm),gam-np.cos(self.cm))
        return lab

    def labAngle2(self) -> float:
        gam = np.sqrt(self.mp*self.me/self.mt/self.mr*self.ep/(self.ep+self.Q*(1+self.mp/self.mt)))
        lab = np.arctan2(np.sin(self.cm),gam+np.cos(self.cm))
        return lab

    def labEnergy(self,mr,me,th) -> float:
        delta = np.sqrt(self.mp*mr*self.ep*np.cos(th)**2 + (me+mr)*(me*self.Q+(me-self.mp)*self.ep))
        if(np.isnan(delta)):
            if(not self.looping):
                print("NaN encountered, Invalid Reaction")
            raise Exception
        fir = np.sqrt(self.mp*mr*self.ep)*np.cos(th)
        e1 = (fir + delta) / (me+mr)
        e2 = (fir - delta) / (me+mr)
        e1 = e1**2
        e2 = e2**2
        gam = np.sqrt(self.mp*mr/self.mt/me*self.ep/(self.ep+self.Q*(1+self.mp/self.mt)))
        arg = np.sin(self.cm)/(gam-np.cos(self.cm))
        der = 1/(1+arg**2)*(gam*np.cos(self.cm)-1)/(gam-np.cos(self.cm))**2
        if (der < 0):
            return e1
        else:
            return e2