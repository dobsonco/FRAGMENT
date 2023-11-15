from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import threading as th
from PIL import ImageTk,Image
from sys import path
import os
import multiprocessing as mp

global sys_path
sys_path = path[0]
global resource_path
resource_path = os.path.join(sys_path,"Resources")
global amu
amu = 931.5
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

        # Creating Reaction Input Frame
        self.reaction_frame = LabelFrame(self.frame, text = "Reaction Info")
        self.reaction_frame.grid(row=0,column=0,sticky="ew",padx=10,pady=5)

        self.mbeam_label = Label(self.reaction_frame, text = "Mass of Beam (amu)")
        self.mbeam_label.grid(row=0,column=0)
        self.mbeam_entry = Entry(self.reaction_frame,textvariable=IntVar(value=12))
        self.mbeam_entry.grid(row=1,column=0)

        self.mtarget_label = Label(self.reaction_frame, text = "Mass of Target (amu)")
        self.mtarget_label.grid(row=0,column=1)
        self.mtarget_entry = Entry(self.reaction_frame,textvariable=IntVar(value=1))
        self.mtarget_entry.grid(row=1,column=1)

        self.beamke_label = Label(self.reaction_frame,text="Beam KE (MeV)")
        self.beamke_label.grid(row=0,column=2)
        self.beamke_entry = Entry(self.reaction_frame,textvariable=IntVar(value=1000))
        self.beamke_entry.grid(row=1,column=2)

        self.mbeamlike_label = Label(self.reaction_frame, text = "Mass of Beamlike (amu)")
        self.mbeamlike_label.grid(row=0,column=3)
        self.mbeamlike_entry = Entry(self.reaction_frame,textvariable=IntVar(value=10))
        self.mbeamlike_entry.grid(row=1,column=3)

        self.mtargetlike_label = Label(self.reaction_frame, text = "Mass of Targetlike (amu)")
        self.mtargetlike_label.grid(row=2,column=0)
        self.mtargetlike_entry = Entry(self.reaction_frame,textvariable=IntVar(value=3))
        self.mtargetlike_entry.grid(row=3,column=0)

        self.comangle_label = Label(self.reaction_frame, text = "Enter CM Angle (deg)")
        self.comangle_label.grid(row=2,column=1)
        self.comangle_entry = Entry(self.reaction_frame,textvariable=IntVar(value=10))
        self.comangle_entry.grid(row=3,column=1)

        self.nreaction_label = Label(self.reaction_frame, text = "# Reactions (thousands)")
        self.nreaction_label.grid(row=2,column=2)
        self.nreaction_entry = Entry(self.reaction_frame,textvariable=IntVar(value=10))
        self.nreaction_entry.grid(row=3,column=2)

        for widget in self.reaction_frame.winfo_children():
           widget.grid_configure(padx=5,pady=5)

        # Creating Dimension Input Frame
        self.dim_frame = LabelFrame(self.frame, text = "Dimensions of Detector")
        self.dim_frame.grid(row=1,column=0,sticky="ew",padx=10,pady=5)

        self.x_dim_label = Label(self.dim_frame, text = "Enter Length (cm)")
        self.x_dim_label.grid(row=0,column=0)
        self.x_dim_entry = Entry(self.dim_frame,textvariable=IntVar(value=100))
        self.x_dim_entry.grid(row=1,column=0)

        self.y_dim_label = Label(self.dim_frame, text = "Enter Radius (cm)")
        self.y_dim_label.grid(row=0,column=1)
        self.y_dim_entry = Entry(self.dim_frame,textvariable=IntVar(value=18))
        self.y_dim_entry.grid(row=1,column=1)

        self.deadzone_label = Label(self.dim_frame, text = "Enter Deadzone (cm)")
        self.deadzone_label.grid(row=0,column=2)
        self.deadzone_entry = Entry(self.dim_frame,textvariable=IntVar(value=6))
        self.deadzone_entry.grid(row=1,column=2)

        self.threshold_label = Label(self.dim_frame, text="Threshold to Detect (cm)")
        self.threshold_label.grid(row=0,column=3)
        self.threshold_entry = Entry(self.dim_frame,textvariable=IntVar(value=3))
        self.threshold_entry.grid(row=1,column=3)

        for widget in self.dim_frame.winfo_children():
            widget.grid_configure(padx=5,pady=5)

        # Create Run Button Frame
        self.button_frame = LabelFrame(self.frame, text = "Reaction Info")
        self.button_frame.grid(row=2,column=0,sticky="ew",padx=10,pady=5)

        self.read_button = Button(self.button_frame,text="Read Inputs",command=self.read_input)
        self.read_button.grid(row=0,column=0)

        self.run_button = Button(self.button_frame,text="Run Sim",command=self.run)
        self.run_button.grid(row=0,column=1)
        self.run_button["state"] = "disabled"

        for widget in self.button_frame.winfo_children():
            widget.grid_configure(padx=5,pady=5)

    def on_x(self):
        self.destroy()

    def read_input(self):
        self.xdim = int(self.x_dim_entry.get())
        self.ydim = int(self.y_dim_entry.get())
        self.dead = int(self.deadzone_entry.get())
        self.threshd = int(self.threshold_entry.get()) + self.dead
        self.ke = int(self.beamke_entry.get())
        self.mp = int(self.mbeam_entry.get())
        self.ep = self.mp * amu + self.ke
        self.mt = int(self.mtarget_entry.get())
        self.et = self.mt * amu
        self.me = int(self.mbeamlike_entry.get())
        self.ee = self.me * amu
        self.mr = int(self.mtargetlike_entry.get())
        self.er = self.mr * amu
        self.cm = int(self.comangle_entry.get()) * (np.pi / 180)
        self.nreactions = int(self.nreaction_entry.get()) * 1000

        if self.threshd >= self.ydim:
            self.errMessage("Value Error", "Detection threshold greater than radius of detector")
            self.toggleRunButton("off")
            return

        if self.run_button["state"] == "disabled":
            self.toggleRunButton("on")

        self.KM.setKinematics(self.mp,self.ep,self.mt,self.et,self.mr,
                        self.er,self.me,self.ee,self.ke,self.cm,
                        self.nreactions,self.xdim,self.ydim,
                        self.dead,self.threshd)
        
    def run(self):
        self.toggleRunButton("off")
        t = th.Thread(self.KM.determineDetected())
        t.start()

    def errMessage(self,errtype: str,message: str):
        messagebox.showwarning(title=errtype,message=message)

    def toggleRunButton(self,state):
        """
        options for state:
        "on" or "off"
        """
        if state == "off":
            self.run_button["state"] = "disabled"
        elif state == "on":
            self.run_button["state"] = "active"
        
    def createFigViewer():
        pass

class Kinematics():
    def __init__(self):
        pass

    def setKinematics(self,mp,ep,mt,et,mr,er,me,ee,ke,cm,nreactions,xdim,ydim,dead,threshd) -> None:
        '''
        Sets Variables for genKinematics
        '''
        self.Q = (ep + et) - (er + ee)
        #print(self.Q)
        self.mp = mp
        #print(self.mp)
        self.ep = ep
        #print(self.ep)
        self.mt = mt
        #print(self.mt)
        self.et = et
        #print(self.et)
        self.mr = mr
        #print(self.mr)
        self.er = er
        #print(self.er)
        self.me = me
        #print(self.me)
        self.ee = ee
        #print(self.ee)
        self.ke = ke
        #print(self.ke)
        self.xdim = xdim
        #print(self.xdim)
        self.ydim = ydim
        #print(self.ydim)
        self.dead = dead
        #print(self.dead)
        self.threshd = threshd
        #print(self.threshd)
        self.cm = cm
        # print(self.cm)
        self.nreactions = nreactions
        # print(self.nreactions)
        self.labA1 = self.labAngle(self.me,self.mr,cm)
        self.labE1 = self.labEnergy(self.me,self.mr,self.labA1,cm)/self.me
        self.labA2 = self.labAngle2(self.mr,self.me,-np.pi+cm)
        self.labE2 = self.labEnergy(self.mr,self.me,self.labA2,-np.pi+cm)/self.mr
        print(self.labA1,self.labE1,self.labA2,self.labE2)

    def determineDetected(self):
        detection = np.zeros((3,self.nreactions))
        for i in range(self.nreactions):
            vz = np.random.uniform(low=0.01,high=self.xdim-0.01)
            detection[2][i] = vz
            y1 = (self.xdim-vz)/np.tan((np.pi/2)-abs(self.labA1))
            y2 = (self.xdim-vz)/np.tan((np.pi/2)-abs(self.labA2))
            if y2 >= self.threshd:
                detection[1][i] = 1.0
            elif y1 >= self.threshd:
                detection[0][i] = 1.0
        self.detectedVert = detection[2][np.logical_or(detection[0]==1.0,detection[1]==1.0)]

        if len(self.detectedVert) <= 0:
            GUI.errMessage("Invalid Reaction","Something went wrong, check reaction info")
            return
        
        GUI.toggleRunButton(state="on")

        p = mp.Process(target=self.createFig)
        p.start()
        
    def createFig(self):
        fig,ax = plt.subplots(nrows=2,ncols=1)
        ax[0].set_xlabel("Vertex of Reaction")
        ax[0].set_ylabel("Counts")
        ax[0].set_title("Number of detections")
        ax[0].set_facecolor('#ADD8E6')
        ax[0].set_axisbelow(True)
        ax[0].yaxis.grid(color='white', linestyle='-')
        ax[0].hist(self.detectedVert,bins=np.arange(min(self.detectedVert),max(self.detectedVert)+0.1,0.2))

        ax[1].set_xlabel("CM Angle")
        ax[1].set_ylabel("Counts")
        ax[1].set_title("Number of detections")
        ax[1].set_facecolor('#ADD8E6')
        ax[1].set_axisbelow(True)
        ax[1].yaxis.grid(color='white', linestyle='-')
        ax[1].hist(self.detectedVert,bins=np.arange(min(self.detectedVert),max(self.detectedVert)+0.1,0.2))
        plt.tight_layout()
        plt.savefig(os.path.join(temp_folder,'fig1.jpg'),format="jpg")
        plt.show()
        plt.cla()
        plt.clf()
        plt.close('all')

    def labAngle(self,me,mr,cm):
        gam = np.sqrt(self.mp*me/self.mt/mr*self.ep/(self.ep+self.Q*(1+self.mp/self.mt)))
        lab = np.arctan2(np.sin(cm),gam-np.cos(cm))
        return lab

    def labAngle2(self,me,mr,cm):
        gam = np.sqrt(self.mp*me/self.mt/mr*self.ep/(self.ep+self.Q*(1+self.mp/self.mt)))
        lab = np.arctan2(np.sin(cm),gam+np.cos(cm))
        return lab

    def labEnergy(self,me,mr,th,cm):
        delta = np.sqrt(self.mp*me*self.ep*np.cos(th)**2 + (mr+me)*(mr*self.Q+(mr-self.mp)*self.ep))
        if(np.isnan(delta)):
            print("NaN encountered, Invalid Reaction")
        fir = np.sqrt(self.mp*me*self.ep)*np.cos(th)
        e1 = (fir + delta) / (mr+me)
        e2 = (fir - delta) / (mr+me)
        e1 = e1**2
        e2 = e2**2
        gam = np.sqrt(self.mp*me/self.mt/mr*self.ep/(self.ep+self.Q*(1+self.mp/self.mt)))
        arg = np.sin(cm)/(gam-np.cos(cm))
        der = 1/(1+arg**2)*(gam*np.cos(cm)-1)/(gam-np.cos(cm))**2
        if (der < 0):
            return e1
        else:
            return e2

if __name__ == '__main__':
    GUI = Window()
    GUI.mainloop()