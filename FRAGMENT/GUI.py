from tkinter import *
from tkinter import messagebox
import numpy as np
from threading import Thread
from PIL import ImageTk
import os
from json import load,dump
from datetime import datetime
from sys import path
from .Kinematics import Kinematics

class Window(Tk):
    def __init__(self):
        super().__init__()
        self.KM = Kinematics()
        self.sys_path = path[0]
        
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
        self.nreaction_entry = Entry(self.reaction_frame,textvariable=IntVar(value=10))
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

        self.run_button = Button(self.button_frame,text="Run Simple Sim",command=self.runSimple)
        self.run_button.grid(row=0,column=0)

        self.runEN_button = Button(self.button_frame,text="Run Energy Sim",command=self.runEN)
        self.runEN_button.grid(row=0,column=1)

        self.info_button = Button(self.button_frame,text="Info",command=self.infoWin)
        self.info_button.grid(row=0,column=2)

        self.import_button = Button(self.button_frame,text="Import Config",command=self.configWin)
        self.import_button.grid(row=0,column=3)

        self.export_button = Button(self.button_frame,text="Export Config",command=self.exportConfig)
        self.export_button.grid(row=0,column=4)

        for widget in self.button_frame.winfo_children():
            widget.grid_configure(padx=5,pady=5)

    def readConfig(self,NAME) -> None:
        try:
            with open(os.path.join(self.sys_path,NAME)) as jsonfile:
                params: dict = load(jsonfile)
                jsonfile.close()
        except:
            raise Exception

        if "Workspace" in params.keys():
            try:
                workspace_params = params["Workspace"]
                self.KM.temp_folder = workspace_params["solutions_path"]
                self.KM.sys_path = workspace_params["workspace_path"]
            except:
                self.errMessage("","Failed to load Workspace Params")

        if "Isotope Info" in params.keys():
            try:
                isotope_params = params["Isotope Info"]

                self.zbeam_entry.delete(0,END)
                self.zbeam_entry.insert(0,isotope_params["zbeam_entry"])

                self.mbeam_entry.delete(0,END)
                self.mbeam_entry.insert(0,isotope_params["mbeam_entry"])
                            
                self.ztarget_entry.delete(0,END)
                self.ztarget_entry.insert(0,isotope_params["ztarget_entry"])

                self.mtarget_entry.delete(0,END)
                self.mtarget_entry.insert(0,isotope_params["mtarget_entry"])

                self.ztargetlike_entry.delete(0,END)
                self.ztargetlike_entry.insert(0,isotope_params["ztargetlike_entry"])

                self.mtargetlike_entry.delete(0,END)
                self.mtargetlike_entry.insert(0,isotope_params["mtargetlike_entry"])
            except:
                self.errMessage("","Failed to load Isotope Params")
                return
        
        if "Reaction Info" in params.keys():
            try:
                reaction_params = params["Reaction Info"]

                self.beamke_entry.delete(0,END)
                self.beamke_entry.insert(0,reaction_params["beamke_entry"])

                self.comangle_entry.delete(0,END)
                self.comangle_entry.insert(0,reaction_params["comangle_entry"])

                self.nreaction_entry.delete(0,END)
                self.nreaction_entry.insert(0,reaction_params["nreaction_entry"])

                self.excitation_entry.delete(0,END)
                self.excitation_entry.insert(0,reaction_params["excitation_entry"])
            except:
                self.errMessage("","Failed to load Reaction Params")
                return

        if "Dimension of Detector" in params.keys():
            try:
                dim_params = params["Dimension of Detector"]

                self.x_dim_entry.delete(0,END)
                self.x_dim_entry.insert(0,dim_params["x_dim_entry"])

                self.y_dim_entry.delete(0,END)
                self.y_dim_entry.insert(0,dim_params["y_dim_entry"])

                self.deadzone_entry.delete(0,END)
                self.deadzone_entry.insert(0,dim_params["deadzone_entry"])

                self.threshold_entry.delete(0,END)
                self.threshold_entry.insert(0,dim_params["threshold_entry"])
            except:
                self.errMessage("","Failed to load Dimension Params")
                print("Failed to load dimensions, proceeding with default")

        return

    def exportConfig(self) -> None:
        try:
            solutions_path = self.KM.temp_folder
            workspace_path = self.KM.sys_path
            Workspace = {
                "solutions_path":solutions_path,
                "workspace_path":workspace_path
            }

            ke  = int(self.beamke_entry.get()) # kinetic energy in MeV/u
            simple_cm = int(self.comangle_entry.get())# in deg
            nreactions = int(self.nreaction_entry.get()) # in thousands
            ex = int(self.excitation_entry.get()) # Excitation in MeV
            ReactionInfo = {
                "beamke_entry": ke,
                "comangle_entry":simple_cm,
                "nreaction_entry":nreactions,
                "excitation_entry":ex
            }

            zp = int(self.zbeam_entry.get())
            mp = int(self.mbeam_entry.get())
            zt = int(self.ztarget_entry.get())
            mt = int(self.mtarget_entry.get())
            zr = int(self.ztargetlike_entry.get())
            mr = int(self.mtargetlike_entry.get())
            IsotopeInfo = {
                "zbeam_entry": zp,
                "mbeam_entry": mp,
                "ztarget_entry": zt,
                "mtarget_entry": mt,
                "ztargetlike_entry": zr,
                "mtargetlike_entry": mr
            }

            xdim = int(self.x_dim_entry.get())
            ydim = int(self.y_dim_entry.get())
            dead = int(self.deadzone_entry.get())
            threshd = int(self.threshold_entry.get())
            Dimensions = {
                "x_dim_entry": xdim,
                "y_dim_entry": ydim,
                "deadzone_entry": dead,
                "threshold_entry": threshd
            }

            #Add solutions dict

            export = {
                "Workspace":Workspace,
                "Isotope Info": IsotopeInfo,
                "Reaction Info": ReactionInfo,
                "Dimension of Detector": Dimensions,
                # Add solution names later
            }

            with open(os.path.join(self.sys_path,f"{datetime.now():%Y_%m_%d-%H_%M}.json"), 'w') as jsonfile:
                dump(export, jsonfile, indent = 4)
                jsonfile.close()

        except:
            self.errMessage("","Failed to export config")
        return

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
            return
        
        if self.KM.dead >= self.KM.ydim:
            self.errMessage("Value Error", "Detection deadzone greater than radius of detector")
            return
        
        if self.KM.zp > self.KM.mp:
            self.errMessage("Value Error", "Projectile Z less than A ")
            return

        if self.KM.zt > self.KM.mt:
            self.errMessage("Value Error", "Target Z less than A ")
            return
        
        if self.KM.zr > self.KM.mr:
            self.errMessage("Value Error", "Targetlike Z less than A ")
            return

        try:
            self.KM.setKinematics()
        except:
            self.errMessage("","Error setting kinematics")
            return

    def runSimple(self) -> None:
        try:
            self.read_input()
            t = Thread(self.KM.determineDetected())
            t.start()
        except:
            self.errMessage("","Error running simple simulation")
        return

    def runEN(self) -> None:
        try:
            self.read_input()
            t = Thread(self.KM.determineEnergy())
            t.start()
        except:
            self.errMessage("","Error running energy simulation")
        return

    def errMessage(self, errtype: str, message: str) -> None:
        messagebox.showerror(title=errtype,message=message)
        return
        
    def infoWin(self) -> None:
        def delete_monitor(self: Window) -> None:
            self.infoWindow.destroy()
            self.info_button.config(state=NORMAL)
            return

        self.infoWindow = Toplevel(master=self)
        self.infoWindow.protocol("WM_DELETE_WINDOW",lambda: delete_monitor(self))
        self.infoWindow.iconphoto(False,ImageTk.PhotoImage(file=os.path.join(self.KM.resource_path,'FRIBlogo.png'),format='png'))
        self.infoWindow.title('What is this?')
        self.infoWindow.resizable(False, False)
        self.info_button.config(state = DISABLED)

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

    def configWin(self) -> None:
        def readConfigAndCloseWin(self: Window) -> None:
            try:
                NAME = str(self.configMessage.get())
                self.readConfig(NAME)
                delete_monitor(self)
            except:
                self.errMessage("","Invalid Filename")
            return
        def delete_monitor(self: Window,) -> None:
            self.import_button.config(state=NORMAL)
            self.configWindow.destroy()
            return

        self.configWindow = Toplevel(master=self)
        self.configWindow.protocol("WM_DELETE_WINDOW",lambda: delete_monitor(self))
        self.configWindow.iconphoto(False,ImageTk.PhotoImage(file=os.path.join(self.KM.resource_path,'FRIBlogo.png'),format='png'))
        self.configWindow.title('Config Entry')
        #self.configWindow.resizable(False, False)
        self.import_button.config(state=DISABLED)

        self.configFrame = Frame(self.configWindow,padx=10,pady=3)
        self.configFrame.pack()

        self.configLabel = LabelFrame(self.configFrame,text="Enter name of config file")
        self.configLabel.grid(row=0,column=0,sticky="ew",padx=10,pady=5)

        self.configMessage = Entry(self.configLabel,textvariable=StringVar())
        self.configMessage.grid(row=1,column=0,sticky="ew")

        self.configBtn = Button(self.configLabel,text="Read Config",command=lambda: readConfigAndCloseWin(self))
        self.configBtn.grid(row=2,column=0)