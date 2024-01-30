from tkinter import *
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from PIL import ImageTk
from sys import path
import os
from multiprocessing import Process
from pandas import read_csv

global sys_path
sys_path = path[0]
global resource_path
resource_path = os.path.join(sys_path,"Resources")
global temp_folder
temp_folder = os.path.join(sys_path,"temp")

class Window(Tk):
    def __init__(self):
        super().__init__()

        self.KM = Kinematics()
        
        self.protocol("WM_DELETE_WINDOW",self.on_x)
        self.title("AT-TPC Sim")
        self.resizable(False, False)
        self.iconphoto(False,ImageTk.PhotoImage(file=os.path.join(resource_path,'FRIBlogo.png'),format='png'))

        self.frame = Frame(self)
        self.frame.pack()

        # Creating Isotope Info Frame
        self.iso_frame = LabelFrame(self.frame, text = "Isotope Info")
        self.iso_frame.grid(row=0,column=0,sticky="ew",padx=10,pady=3)

        self.mbeam_label = Label(self.iso_frame, text = "Z and A of Beam")
        self.mbeam_label.grid(row=0,column=0)
        self.zbeam_entry = Entry(self.iso_frame,textvariable=IntVar(value=4))
        self.zbeam_entry.grid(row=1,column=0)
        self.mbeam_entry = Entry(self.iso_frame,textvariable=IntVar(value=10))
        self.mbeam_entry.grid(row=2,column=0)

        self.mtarget_label = Label(self.iso_frame, text = "Z and A of Target")
        self.mtarget_label.grid(row=0,column=1)
        self.ztarget_entry = Entry(self.iso_frame,textvariable=IntVar(value=1))
        self.ztarget_entry.grid(row=1,column=1)
        self.mtarget_entry = Entry(self.iso_frame,textvariable=IntVar(value=1))
        self.mtarget_entry.grid(row=2,column=1)

        self.mtargetlike_label = Label(self.iso_frame, text = "Z and A of Targetlike")
        self.mtargetlike_label.grid(row=0,column=2)
        self.ztargetlike_entry = Entry(self.iso_frame,textvariable=IntVar(value=1))
        self.ztargetlike_entry.grid(row=1,column=2)
        self.mtargetlike_entry = Entry(self.iso_frame,textvariable=IntVar(value=1))
        self.mtargetlike_entry.grid(row=2,column=2)

        for widget in self.iso_frame.winfo_children():
           widget.grid_configure(padx=5,pady=5)

        # Creating Reaction Info Frame
        self.reaction_frame = LabelFrame(self.frame, text = "Reaction Info")
        self.reaction_frame.grid(row=1,column=0,sticky="ew",padx=10,pady=3)

        self.beamke_label = Label(self.reaction_frame,text="Beam KE (MeV/u)")
        self.beamke_label.grid(row=0,column=0)
        self.beamke_entry = Entry(self.reaction_frame,textvariable=IntVar(value=10))
        self.beamke_entry.grid(row=1,column=0)

        self.comangle_label = Label(self.reaction_frame, text = "Enter CM Angle (deg)")
        self.comangle_label.grid(row=0,column=1)
        self.comangle_entry = Entry(self.reaction_frame,textvariable=IntVar(value=45))
        self.comangle_entry.grid(row=1,column=1)

        self.nreaction_label = Label(self.reaction_frame, text = "# Reactions (thousands)")
        self.nreaction_label.grid(row=0,column=2)
        self.nreaction_entry = Entry(self.reaction_frame,textvariable=IntVar(value=10))
        self.nreaction_entry.grid(row=1,column=2)

        self.excitation_label = Label(self.reaction_frame, text = "Excitation (MeV)")
        self.excitation_label.grid(row=2,column=0)
        self.excitation_entry = Entry(self.reaction_frame,textvariable=DoubleVar(value=0))
        self.excitation_entry.grid(row=3,column=0)

        # self.vertex_label = Label(self.reaction_frame, text = "Vertex of Reaction (cm)")
        # self.vertex_label.grid(row=2,column=1)
        # self.vertex_entry = Entry(self.reaction_frame,textvariable=DoubleVar(value=50))
        # self.vertex_entry.grid(row=3,column=1)

        for widget in self.reaction_frame.winfo_children():
           widget.grid_configure(padx=5,pady=5)

        # Creating Dimension Input Frame
        self.dim_frame = LabelFrame(self.frame, text = "Dimensions of Detector")
        self.dim_frame.grid(row=2,column=0,sticky="ew",padx=10,pady=5)

        self.x_dim_label = Label(self.dim_frame, text = "Enter Length (cm)")
        self.x_dim_label.grid(row=0,column=0)
        self.x_dim_entry = Entry(self.dim_frame,textvariable=IntVar(value=100))
        self.x_dim_entry.grid(row=1,column=0)

        self.y_dim_label = Label(self.dim_frame, text = "Enter Radius (cm)")
        self.y_dim_label.grid(row=0,column=1)
        self.y_dim_entry = Entry(self.dim_frame,textvariable=IntVar(value=28))
        self.y_dim_entry.grid(row=1,column=1)

        self.deadzone_label = Label(self.dim_frame, text = "Enter Deadzone (cm)")
        self.deadzone_label.grid(row=0,column=2)
        self.deadzone_entry = Entry(self.dim_frame,textvariable=IntVar(value=3))
        self.deadzone_entry.grid(row=1,column=2)

        self.threshold_label = Label(self.dim_frame, text="Threshold to Detect (cm)")
        self.threshold_label.grid(row=2,column=0)
        self.threshold_entry = Entry(self.dim_frame,textvariable=IntVar(value=6))
        self.threshold_entry.grid(row=3,column=0)

        for widget in self.dim_frame.winfo_children():
            widget.grid_configure(padx=5,pady=5)

        # Create Run Button Frame
        self.button_frame = LabelFrame(self.frame, text = "Control Panel")
        self.button_frame.grid(row=3,column=0,sticky="ew",padx=10,pady=5)

        self.read_button = Button(self.button_frame,text="Read Inputs",command=self.read_input)
        self.read_button.grid(row=0,column=0)

        self.run_button = Button(self.button_frame,text="Run Regular Sim",command=self.run)
        self.run_button.grid(row=0,column=1)
        self.run_button["state"] = "disabled"

        self.runEN_button = Button(self.button_frame,text="Run Energy Sim",command=self.runEN)
        self.runEN_button.grid(row=0,column=2)
        self.runEN_button["state"] = "disabled"

        self.info_button = Button(self.button_frame,text="Info",command=self.infoWin)
        self.info_button.grid(row=0,column=3)

        for widget in self.button_frame.winfo_children():
            widget.grid_configure(padx=5,pady=5)

    def on_x(self) -> None:
        self.destroy()

    def read_input(self) -> None:
        self.KM.xdim = int(self.x_dim_entry.get())
        self.KM.ydim = int(self.y_dim_entry.get())
        self.KM.dead = int(self.deadzone_entry.get())
        self.KM.threshd = float(self.threshold_entry.get())
        self.KM.zp = int(self.zbeam_entry.get())
        self.KM.mp = int(self.mbeam_entry.get())
        self.KM.ke  = float(self.beamke_entry.get()) # kinetic energy in MeV/u
        self.KM.ep = self.KM.ke*self.KM.mp # kinetic energy in MeV
        self.KM.zt = int(self.ztarget_entry.get())
        self.KM.mt = int(self.mtarget_entry.get())
        self.KM.et = 0 # target is at rest
        self.KM.zr = int(self.ztargetlike_entry.get())
        self.KM.mr = int(self.mtargetlike_entry.get())
        self.KM.ee = 0
        self.KM.ze = self.KM.zp + self.KM.zt - self.KM.zr # Conservation of Z
        self.KM.me = self.KM.mp + self.KM.mt - self.KM.mr # Conservation of A
        self.KM.er = 0
        self.KM.cm = float(self.comangle_entry.get()) * (np.pi / 180) # in rad
        self.KM.nreactions = int(self.nreaction_entry.get()) * 1000
        self.KM.ex = float(self.excitation_entry.get()) # Excitation in MeV
#        self.KM.defaultVertex = float(self.vertex_entry.get()) # Vertex of Raction for energy sim with fixed vertex

        if self.KM.threshd >= self.KM.ydim:
            self.errMessage("Value Error", "Detection threshold greater than radius of detector")
            self.toggleRunButtons("off")
            return
        
        if self.KM.dead >= self.KM.ydim:
            self.errMessage("Value Error", "Detection deadzone greater than radius of detector")
            self.toggleRunButtons("off")
            return
        
        if self.KM.zp > self.KM.mp:
            self.errMessage("Value Error", "Projectile Z less than A ")
            self.toggleRunButtons("off")
            return

        if self.KM.zt > self.KM.mt:
            self.errMessage("Value Error", "Target Z less than A ")
            self.toggleRunButtons("off")
            return
        
        if self.KM.zr > self.KM.mr:
            self.errMessage("Value Error", "Targetlike Z less than A ")
            self.toggleRunButtons("off")
            return

        self.KM.setKinematics()
        
        if self.run_button["state"] == "disabled":
            self.toggleRunButtons("on")

    def run(self) -> None:
        self.toggleRunButtons("off")
        t = Thread(self.KM.determineDetected())
        t.start()

    def runEN(self) -> None:
        self.toggleRunButtons("off")
        t = Thread(self.KM.determineEnergy())
        t.start()

    def errMessage(self, errtype: str, message: str) -> None:
        messagebox.showwarning(title=errtype,message=message)

    def toggleRunButtons(self,state) -> None:
        """
        options for state:
        "on" or "off"
        """
        if state == "off":
            self.run_button["state"] = "disabled"
            self.runEN_button["state"] = "disabled"
        elif state == "on":
            self.run_button["state"] = "active"
            self.runEN_button["state"] = "active"
        
    def infoWin(self) -> None:
        def delete_monitor(self: GUI) -> None:
            self.infoWindow.destroy()
            self.info_button['state'] = 'active'

        self.infoWindow = Toplevel(master=self)
        self.infoWindow.protocol("WM_DELETE_WINDOW",lambda: delete_monitor(self))
        self.infoWindow.iconphoto(False,ImageTk.PhotoImage(file=os.path.join(resource_path,'FRIBlogo.png'),format='png'))
        self.infoWindow.title('What is this?')
        self.infoWindow.resizable(False, False)
        self.info_button['state'] = 'disabled'

        self.infoFrame = Frame(self.infoWindow)
        self.infoFrame.pack()

        self.infoLabelReaction = LabelFrame(self.infoFrame,text="Reaction Frame Info")
        self.infoLabelReaction.grid(row=0,column=0,sticky="ew",padx=10,pady=5)

        self.infoMessageReaction = Message(self.infoLabelReaction,text='''In the reaction frame, you can enter the Proton Count (Z) and Mass (A) of the Beam, the Z and A of the Target, the Z and A of the Targetlike product, the kinetic energy of the beam particles (MeV/u), the center of mass angle (deg), and the number of reactions to generate''',aspect=700)
        self.infoMessageReaction.grid(column=0,row=0,sticky="ew")

        self.infoLabelDims = LabelFrame(self.infoFrame,text="Dimension Info")
        self.infoLabelDims.grid(row=1,column=0,sticky="ew",padx=10,pady=5)

        self.infoMessageReaction = Message(self.infoLabelDims,text='''In the Dimensions frame, you can enter the Length of the detector (X dimension), the center of mass angle (cm) of the reaction, the deadzone at the center of the detector (for more info, look at the design for the AT-TPC), and the threshold for detection (this is the distance outside of the deadzone required to classify the particle). Particles that do not not pass the threshold but enter the deadzone will be picked up by the zero angle detector.''',aspect=700)
        self.infoMessageReaction.grid(column=0,row=0,sticky="ew")

class Kinematics:
    def __init__(self):
        self.mexcess = read_csv(os.path.join(resource_path,'mexcess.csv')).to_numpy()
        pass

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
            GUI.errMessage("Missing Mass Excess","Missing mass excess for selected nuclei, proceeding without exact numbers")
            mexp=0
            mext=0
            mexr=0
            mexe=0
        
        self.mp = (self.mp*amu + mexp) / amu # convert mass to amu units
        self.mt = (self.mt*amu + mext) / amu # convert mass to amu units
        self.mr = (self.mr*amu + mexr) / amu # convert mass to amu units
        self.me = (self.me*amu + mexe) / amu # convert mass to amu units
        self.Q = (self.mp + self.mt - self.mr - self.me) * amu - self.ex # in MeV

        self.labA1 = self.labAngle()
        self.labE1 = self.labEnergy(self.mr,self.me,self.labA1)/self.mr
        self.labA2 = self.labAngle2()
        self.labE2 = self.labEnergy(self.me,self.mr,self.labA2)/self.me # Swap mr and me for the beam-like
        print("A1:",self.labA1*180/np.pi,"E1:",self.labE1,"A2:",self.labA2*180/np.pi,"E2:",self.labE2,"Q:",self.Q)

    def determineDetected(self) -> None:
        self.detectedVert = []
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
                self.detectedVert.append(vz)

        if len(self.detectedVert) <= 0:
            GUI.errMessage("Invalid Reaction","Something went wrong, check reaction info")
            return
        
        self.detection2 = []
        self.detection3 = []
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
                self.detection2.append(cm*180/np.pi)
                self.detection3.append(vz)

        if len(self.detection2) <= 0:
            GUI.errMessage(ValueError,"No particles detected")

        GUI.toggleRunButtons(state="on")

        p = Process(target=self.createFig)
        p.start()

    def determineEnergy(self) -> None:
        vz = self.xdim / 2
        self.Energy1 = []
        self.Cm1 = []
        for _ in range(self.nreactions):
            self.cm = np.random.uniform(0,np.pi)
            A1 = self.labAngle()
            A2 = self.labAngle2()
            if self.labA1 < np.pi/2:
                y1 = (self.xdim-vz)*np.tan(self.labA1)
            else:
                y1 = vz*np.tan(self.labA1)
            if self.labA2 < np.pi/2:
                y2 = (self.xdim-vz)*np.tan(self.labA2)
            else:
                y2 = vz*np.tan(self.labA2)
            if y1 >= self.dead and y2 <= self.threshd:
                self.Cm1.extend([self.cm,self.cm])
                self.Energy1.extend([self.labEnergy(self.mr,self.me,A1),self.labEnergy(self.me,self.mr,A2)])

        GUI.toggleRunButtons(state="on")

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
        ax[0].hist(self.detectedVert,bins=100,range=(0,self.xdim))

        ax[1].set_xlabel("CM Angle (deg)")
        ax[1].set_ylabel("Counts")
        ax[1].set_title(f"Number of detections with random cm and vz")
        ax[1].set_facecolor('#ADD8E6')
        ax[1].set_axisbelow(True)
        ax[1].yaxis.grid(color='white', linestyle='-')
        ax[1].hist(self.detection2,bins=180,range=(0,180))

        ax[2].set_xlabel("Vertex")
        ax[2].set_ylabel("Counts")
        ax[2].set_title(f"Number of detections with random cm and vz")
        ax[2].set_facecolor('#ADD8E6')
        ax[2].set_axisbelow(True)
        ax[2].yaxis.grid(color='white', linestyle='-')
        ax[2].hist(self.detection3,bins=100,range=(0,self.xdim))

        plt.tight_layout()
        plt.savefig(os.path.join(temp_folder,'fig1.jpg'),format="jpg")
        plt.show()
        plt.cla()
        plt.clf()
        plt.close('all')

    def createENFig(self) -> None:
        fig,ax = plt.subplots(nrows=1,ncols=1,dpi=150)
        ax.set_xlabel("CM angle")
        ax.set_ylabel("Energy (MeV)")
        ax.set_title(f"Energy for fixed vz = {self.xdim / 2} and random CM")
        ax.set_facecolor('#ADD8E6')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='white', linestyle='-')
        ax.scatter(x=self.Cm1,y=self.Energy1,s=1)

        plt.tight_layout()
        plt.savefig(os.path.join(temp_folder,'fig1.jpg'),format="jpg")
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
            print("NaN encountered, Invalid Reaction")
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

if __name__ == '__main__':
    GUI = Window()
    GUI.mainloop()