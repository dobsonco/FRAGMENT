from tkinter import *
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import threading as th
from PIL import ImageTk,Image
from sys import path
import os

global sys_path
sys_path = path[0]
global resource_path
resource_path = os.path.join(sys_path,"Resources")

class Window(Tk):
    def __init__(self):
        super().__init__()
        
        self.protocol("WM_DELETE_WINDOW",self.on_x)
        self.title("AT-TPC Sim")
        self.resizable(False, False)
        self.iconphoto(False,ImageTk.PhotoImage(file=os.path.join(resource_path,'FRIBlogo.png'),format='png'))

        # Creating Dimension Input Frame
        self.frame = Frame(self)
        self.frame.pack()

        self.dim_frame = LabelFrame(self.frame, text = "Dimensions of Detector")
        self.dim_frame.grid(row=0,column=0,padx=20,pady=10)

        self.x_dim_label = Label(self.dim_frame, text = "Enter Xdim (cm)")
        self.x_dim_label.grid(row=0,column=0)
        self.x_dim_entry = Entry(self.dim_frame)
        self.x_dim_entry.grid(row=1,column=0,padx=5,pady=5)
        self.x_dim_val = IntVar()

        self.y_dim_label = Label(self.dim_frame, text = "Enter Ydim (cm)")
        self.y_dim_label.grid(row=0,column=1)
        self.y_dim_entry = Entry(self.dim_frame)
        self.y_dim_entry.grid(row=1,column=1,padx=5,pady=5)
        self.y_dim_val = IntVar()

        self.deadzone_label = Label(self.dim_frame, text = "Enter Deadzone (cm)")
        self.deadzone_label.grid(row=0,column=2)
        self.deadzone_entry = Entry(self.dim_frame)
        self.deadzone_entry.grid(row=1,column=2,padx=5,pady=5)
        self.deadzone_val = IntVar()

        self.threshold_label = Label(self.dim_frame, text="Threshold to Detect (cm)")
        self.threshold_label.grid(row=0,column=3)
        self.threshold_entry = Entry(self.dim_frame)
        self.threshold_entry.grid(row=1,column=3,padx=5,pady=5)
        self.threshold_val = IntVar()

        # Creating Reaction Input Frame
        self.reaction_frame = LabelFrame(self.frame, text = "Reaction Info")
        self.reaction_frame.grid(row=1,column=0,padx=5,pady=5)

        self.mbeam_label = Label(self.reaction_frame, text = "Mass of Beam (amu)")
        self.mbeam_label.grid(row=0,column=0,padx=5,pady=5)
        self.mbeam_entry = Entry(self.reaction_frame)
        self.mbeam_entry.grid(row=1,column=0,padx=5,pady=5)
        self.mbeam_val = IntVar()

        self.mbeam_label = Label(self.reaction_frame, text = "Mass of Beam (amu)")
        self.mbeam_label.grid(row=0,column=1,padx=5,pady=5)
        self.mbeam_entry = Entry(self.reaction_frame)
        self.mbeam_entry.grid(row=1,column=1,padx=5,pady=5)
        self.mbeam_val = IntVar()

        self.mtarget_label = Label(self.reaction_frame, text = "Mass of Target (amu)")
        self.mtarget_label.grid(row=0,column=2,padx=5,pady=5)
        self.mtarget_entry = Entry(self.reaction_frame)
        self.mtarget_entry.grid(row=1,column=2,padx=5,pady=5)
        self.mtarget_val = IntVar()

        self.beamke_label = Label(self.reaction_frame,text="Mass of Target (amu)")
        self.beamke_label.grid(row=0,column=3)
        self.beamke_entry = Entry(self.reaction_frame)
        self.beamke_entry.grid(row=1,column=3,padx=5,pady=5)
        self.beamke_val = IntVar()

        

    def on_x(self):
        self.destroy()

GUI = Window()
GUI.mainloop()