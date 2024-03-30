from tkinter import *
from tkinter import messagebox
import numpy as np
from threading import Thread
from PIL import ImageTk
import os
from .Kinematics import Kinematics

class Window(Tk):
    def __init__(self):
        super().__init__()
        self.KM = Kinematics()
        
        self.protocol("WM_DELETE_WINDOW",lambda:self.destroy())
        self.title("AT-TPC Sim")
        self.resizable(False, False)
        self.iconphoto(False,ImageTk.PhotoImage(file=os.path.join(self.KM.resource_path,'FRIBlogo.png'),format='png'))

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
        self.nreaction_entry = Entry(self.reaction_frame,textvariable=IntVar(value=1))
        self.nreaction_entry.grid(row=1,column=2)

        self.excitation_label = Label(self.reaction_frame, text = "Excitation (MeV)")
        self.excitation_label.grid(row=2,column=0)
        self.excitation_entry = Entry(self.reaction_frame,textvariable=DoubleVar(value=0))
        self.excitation_entry.grid(row=3,column=0)

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

        # self.read_button = Button(self.button_frame,text="Read Inputs",command=self.read_input)
        # self.read_button.grid(row=0,column=0)

        self.run_button = Button(self.button_frame,text="Run Regular Sim",command=self.run)
        self.run_button.grid(row=0,column=0)
        # self.run_button["state"] = "disabled"

        self.runEN_button = Button(self.button_frame,text="Run Energy Sim",command=self.runEN)
        self.runEN_button.grid(row=0,column=1)
        # self.runEN_button["state"] = "disabled"

        self.info_button = Button(self.button_frame,text="Info",command=self.infoWin)
        self.info_button.grid(row=0,column=2)

        for widget in self.button_frame.winfo_children():
            widget.grid_configure(padx=5,pady=5)

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
        #self.KM.et = 0 # target is at rest
        self.KM.zr = int(self.ztargetlike_entry.get())
        self.KM.mr = int(self.mtargetlike_entry.get())
        #self.KM.ee = 0
        self.KM.ze = self.KM.zp + self.KM.zt - self.KM.zr # Conservation of Z
        self.KM.me = self.KM.mp + self.KM.mt - self.KM.mr # Conservation of A
        #self.KM.er = 0
        self.KM.simple_cm = float(self.comangle_entry.get()) * (np.pi / 180) # in rad
        self.KM.nreactions = int(self.nreaction_entry.get()) * 1000
        self.KM.ex = float(self.excitation_entry.get()) # Excitation in MeV
        #self.KM.defaultVertex = float(self.vertex_entry.get()) # Vertex of Raction for energy sim with fixed vertex

        if self.KM.threshd >= self.KM.ydim:
            self.errMessage("Value Error", "Detection threshold greater than radius of detector")
            # self.toggleRunButtons("off")
            return
        
        if self.KM.dead >= self.KM.ydim:
            self.errMessage("Value Error", "Detection deadzone greater than radius of detector")
            # self.toggleRunButtons("off")
            return
        
        if self.KM.zp > self.KM.mp:
            self.errMessage("Value Error", "Projectile Z less than A ")
            # self.toggleRunButtons("off")
            return

        if self.KM.zt > self.KM.mt:
            self.errMessage("Value Error", "Target Z less than A ")
            # self.toggleRunButtons("off")
            return
        
        if self.KM.zr > self.KM.mr:
            self.errMessage("Value Error", "Targetlike Z less than A ")
            # self.toggleRunButtons("off")
            return

        try:
            self.KM.setKinematics()
        except:
            self.errMessage("","Error setting kinematics")
            return
        
        if self.run_button["state"] == "disabled":
            self.toggleRunButtons("on")

    def run(self) -> None:
        # self.toggleRunButtons("off")
        try:
            self.read_input()
            t = Thread(self.KM.determineDetected())
            t.start()
        except:
            self.errMessage("","Error running simple simulation")

    def runEN(self) -> None:
        # self.toggleRunButtons("off")
        self.read_input()
        t = Thread(self.KM.determineEnergy())
        t.start()

    def errMessage(self, errtype: str, message: str) -> None:
        messagebox.showerror(title=errtype,message=message)

    # def toggleRunButtons(self,state) -> None:
    #     """
    #     options for state:
    #     "on" or "off"
    #     """
    #     if state == "off":
    #         self.run_button["state"] = "disabled"
    #         self.runEN_button["state"] = "disabled"
    #     elif state == "on":
    #         self.run_button["state"] = "active"
    #         self.runEN_button["state"] = "active"
        
    def infoWin(self) -> None:
        def delete_monitor(self: Window) -> None:
            self.infoWindow.destroy()
            self.info_button['state'] = 'active'

        self.infoWindow = Toplevel(master=self)
        self.infoWindow.protocol("WM_DELETE_WINDOW",lambda: delete_monitor(self))
        self.infoWindow.iconphoto(False,ImageTk.PhotoImage(file=os.path.join(self.KM.resource_path,'FRIBlogo.png'),format='png'))
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