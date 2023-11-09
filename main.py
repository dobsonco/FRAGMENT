from tkinter import *
from tkinter import ttk
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

class Kinematics():
    def __init__(self, ee, er, ep, et, nreactions=100):
        self.Q = (ep + et) - (er + ee)

        self.reactionArr = np.zeros((5,nreactions),dtype=float)

        self.reactionArr[0] = np.random.uniform(0,1,nreactions)*np.pi

    def genKinematics(self,mp,mt,me,mr,ep) -> np.ndarray:
        for i,cm in enumerate(self.reactionArr[0]):
            self.reactionArr[1][i] = self.labAngle(mp,mt,me,mr,ep,self.Q,cm)
            self.reactionArr[2][i] = self.labEnergy(mp,mt,me,mr,ep,self.Q,self.reactionArr[1][i],cm)/me
            self.reactionArr[3][i] = self.labAngle2(mp,mt,mr,me,ep,self.Q,-np.pi+cm)
            self.reactionArr[4][i] = self.labEnergy(mp,mt,mr,me,ep,self.Q,self.reactionArr[3][i],cm)/mr
        print(self.reactionArr)

    def labAngle(self,mp,mt,me,mr,ep,Q,cm):
        gam = np.sqrt(mp*me/mt/mr*ep/(ep+Q*(1+mp/mt)))
        lab = np.arctan2(np.sin(cm),gam-np.cos(cm))
        return lab

    def labAngle2(self,mp,mt,me,mr,ep,Q,cm):
        gam = np.sqrt(mp*me/mt/mr*ep/(ep+Q*(1+mp/mt)))
        lab = np.arctan2(np.sin(cm),gam+np.cos(cm))
        return lab

    def labEnergy(self,mp,mt,me,mr,ep,Q,th,cm):
        delta = np.sqrt(mp*me*ep*np.cos(th)**2 + (mr+me)*(mr*self.Q+(mr-mp)*ep))
        if(np.isnan(delta)):
            print("NaN encountered, Invalid Reaction")
        fir = np.sqrt(mp*me*ep)*np.cos(th)
        e1 = (fir + delta) / (mr+me)
        e2 = (fir - delta) / (mr+me)
        e1 = e1**2
        e2 = e2**2
        gam = np.sqrt(mp*me/mt/mr*ep/(ep+self.Q*(1+mp/mt)))
        arg = np.sin(cm)/(gam-np.cos(cm))
        der = 1/(1+arg**2)*(gam*np.cos(cm)-1)/(gam-np.cos(cm))**2
        if (der < 0):
            return e1
        else:
            return e2

class Window(Tk):
    def __init__(self):
        super().__init__()
        
        self.protocol("WM_DELETE_WINDOW",self.on_x)
        self.title("AT-TPC Sim")
        self.resizable(False, False)
        self.iconphoto(False,ImageTk.PhotoImage(file=os.path.join(resource_path,'FRIBlogo.png'),format='png'))

        self.frame = Frame(self)
        self.frame.pack()

        # Creating Reaction Input Frame
        self.reaction_frame = LabelFrame(self.frame, text = "Reaction Info")
        self.reaction_frame.grid(row=0,column=0,padx=5,pady=5,sticky="ew")

        self.mbeam_label = Label(self.reaction_frame, text = "Mass of Beam (amu)")
        self.mbeam_label.grid(row=0,column=0,padx=5,pady=5)
        self.mbeam_entry = Entry(self.reaction_frame,textvariable=IntVar(value=12))
        self.mbeam_entry.grid(row=1,column=0,padx=5,pady=5)
        self.mbeam_val = IntVar()

        self.mtarget_label = Label(self.reaction_frame, text = "Mass of Target (amu)")
        self.mtarget_label.grid(row=0,column=1,padx=5,pady=5)
        self.mtarget_entry = Entry(self.reaction_frame,textvariable=IntVar(value=1))
        self.mtarget_entry.grid(row=1,column=1,padx=5,pady=5)
        self.mtarget_val = IntVar()

        self.beamke_label = Label(self.reaction_frame,text="Beam KE (MeV)")
        self.beamke_label.grid(row=0,column=2)
        self.beamke_entry = Entry(self.reaction_frame,textvariable=IntVar(value=1000))
        self.beamke_entry.grid(row=1,column=2,padx=5,pady=5)
        self.beamke_val = IntVar()

        self.mbeamlike_label = Label(self.reaction_frame, text = "Mass of Beamlike (amu)")
        self.mbeamlike_label.grid(row=0,column=3,padx=5,pady=5)
        self.mbeamlike_entry = Entry(self.reaction_frame,textvariable=IntVar(value=10))
        self.mbeamlike_entry.grid(row=1,column=3,padx=5,pady=5)
        self.mbeamlike_val = IntVar()

        self.mtargetlike_label = Label(self.reaction_frame, text = "Mass of Targetlike (amu)")
        self.mtargetlike_label.grid(row=2,column=0,padx=5,pady=5)
        self.mtargetlike_entry = Entry(self.reaction_frame,textvariable=IntVar(value=3))
        self.mtargetlike_entry.grid(row=3,column=0,padx=5,pady=5)
        self.mtargetlike_val = IntVar()

        self.comangle_label = Label(self.reaction_frame, text = "Enter CM Angle (deg)")
        self.comangle_label.grid(row=2,column=1,padx=5,pady=5)
        self.comangle_entry = Entry(self.reaction_frame,textvariable=IntVar(value=10))
        self.comangle_entry.grid(row=3,column=1,padx=5,pady=5)
        self.comangle_val = IntVar()

        # Creating Dimension Input Frame
        self.dim_frame = LabelFrame(self.frame, text = "Dimensions of Detector")
        self.dim_frame.grid(row=1,column=0,padx=5,pady=5,sticky="ew")

        self.x_dim_label = Label(self.dim_frame, text = "Enter Xdim (cm)")
        self.x_dim_label.grid(row=0,column=0)
        self.x_dim_entry = Entry(self.dim_frame,textvariable=IntVar(value=100))
        self.x_dim_entry.grid(row=1,column=0,padx=5,pady=5)
        self.x_dim_val = IntVar()

        self.y_dim_label = Label(self.dim_frame, text = "Enter Ydim (cm)")
        self.y_dim_label.grid(row=0,column=1)
        self.y_dim_entry = Entry(self.dim_frame,textvariable=IntVar(value=36))
        self.y_dim_entry.grid(row=1,column=1,padx=5,pady=5)
        self.y_dim_val = IntVar()

        self.deadzone_label = Label(self.dim_frame, text = "Enter Deadzone (cm)")
        self.deadzone_label.grid(row=0,column=2)
        self.deadzone_entry = Entry(self.dim_frame,textvariable=IntVar(value=6))
        self.deadzone_entry.grid(row=1,column=2,padx=5,pady=5)
        self.deadzone_val = IntVar()

        self.threshold_label = Label(self.dim_frame, text="Threshold to Detect (cm)")
        self.threshold_label.grid(row=0,column=3)
        self.threshold_entry = Entry(self.dim_frame,textvariable=IntVar(value=3))
        self.threshold_entry.grid(row=1,column=3,padx=5,pady=5)
        self.threshold_val = IntVar()

        # Create Run Button Frame
        self.button_frame = LabelFrame(self.frame, text = "Reaction Info")
        self.button_frame.grid(row=2,column=0,padx=5,pady=5,sticky="ew")

        self.run_button = Button(self.button_frame,text="Read Inputs",command=self.read_input)
        self.run_button.pack(padx=5,pady=5)

    def on_x(self):
        self.destroy()

    def read_input(self):
        self.x_dim_val = self.x_dim_entry.get()
        print(self.x_dim_val)

        self.y_dim_val = self.y_dim_entry.get()
        print(self.y_dim_val)

        self.deadzone_val = self.deadzone_entry.get()
        print(self.deadzone_val)

        self.threshold_val = self.threshold_entry.get()
        print(self.threshold_val)

        self.mbeam_val = self.mbeam_entry.get()
        print(self.mbeam_val)

        self.mtarget_val = self.mtarget_entry.get()
        print(self.mtarget_val)

        self.beamke_val = self.beamke_entry.get()
        print(self.beamke_val)

        self.mbeamlike_val = self.mbeamlike_entry.get()
        print(self.mbeamlike_val)

        self.mtargetlike_val = self.mtargetlike_entry.get()
        print(self.mtargetlike_val)

        self.comangle_val = self.comangle_entry.get()
        print(self.comangle_val)

if __name__ == '__main__':
    GUI = Window()
    GUI.mainloop()