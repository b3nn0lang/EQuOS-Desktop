# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:25:46 2026

@author: benno
"""
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from QS_data_sender import Send_esr_measurement, Send_rabi_measurement, Send_t1_measurement, User_ESR_input, Qupyt_controller, User_Rabi_input, User_T1_input
from QS_data_reader import QS_data_reader
from Equos_logic import create_file_name, parse_file_folder, calculate_t1_averages, ECQS_cache
from Data_evaluator import qs_data_displayer, ec_data_displayer, ecqs_data_displayer
import os
import matplotlib.pyplot as plt
from Potentiostat import Potentiostat
import numpy as np
import gc

max_db = 8 
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.current_measurement_method = None
        self.esr = User_ESR_input()
        self.rabi = User_Rabi_input()
        self.t1 = User_T1_input()
        self.ecqs_cache = ECQS_cache()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Configure>", self.on_resize)
        self.focus_force()
        self.after(0,self.layout)
        
    def layout(self):
        self.title("EQuOS")
        self.cancel_requested = False
        self.live_update_job = None
        self.state("zoomed")
        self.iconbitmap("logo/EQuOS_icon.ico")       # .ico file
        self.photo = tk.PhotoImage(file="logo/EQuOS_logo.png")
        self.logo = tk.Label(self, image=self.photo)
        self.logo.pack()
        tk.Button(self, text="📁 change directory", command=self.on_folder_send).pack()
        self.focus_force()
        self.frames = {}
        self.fitting_frames = {}
        
       # 1. Create a canvas and a scrollbar
        canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        # 2. Create the actual frame inside the canvas
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        # 4. Put the frame in the canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))
        # 5. Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        #measure
        measure = tk.Frame(scrollable_frame)
        self.frames["measure"] = measure
                #boot
        
        self.boot = tk.Frame(measure)
        self.boot_label = tk.Label(self.boot, text="Booting Qupyt...")
        self.boot_label.pack()
        self.boot_progress = ttk.Progressbar(self.boot, length=300, mode="indeterminate")
        self.boot_progress.pack()
        self.boot.grid(row=0, column=0)
        
        
        
        self.measurement_frame = tk.Frame(measure, bd=2, relief="groove")
        self.measurement_frame.grid(row=0,column=0)
        
        self.method_choice_frame = tk.Frame(self.measurement_frame)
        self.method_choice_frame.pack()
        
        for col in range(3):
            self.method_choice_frame.grid_columnconfigure(col, weight=1)
        self.measurement_methods = {}
        #Method Buttons
        self.measurement_choice = tk.StringVar(value="ESR")  # default selection
        
        esr_method_button =tk.Radiobutton(self.method_choice_frame,text="ESR",variable=self.measurement_choice,value="ESR",command=self.change_method)                
        esr_method_button.grid(row=0, column=0,sticky="w")
        rabi_method_button = tk.Radiobutton(self.method_choice_frame,text="Rabi",variable=self.measurement_choice,value="Rabi",command=self.change_method)   
        rabi_method_button.grid(row=0,column=1,sticky="w")
        t1_method_button = tk.Radiobutton(self.method_choice_frame,text="T1",variable=self.measurement_choice,value="T1",command=self.change_method) 
        t1_method_button.grid(row=0,column=2,sticky="w")
        cv_method_button = tk.Radiobutton(self.method_choice_frame,text="CV",variable=self.measurement_choice,value="CV",command=self.change_method)
        cv_method_button.grid(row=1,column=0,sticky="w")
        swv_method_button = tk.Radiobutton(self.method_choice_frame,text="SWV",variable=self.measurement_choice,value="SWV",command=self.change_method)
        swv_method_button.grid(row=1,column=1,sticky="w")
        lsv_method_button = tk.Radiobutton(self.method_choice_frame,text="LSV",variable=self.measurement_choice,value="LSV",command=self.change_method)
        lsv_method_button.grid(row=1,column=2,sticky="w")
        cv_t1_method_button = tk.Radiobutton(self.method_choice_frame,text="CV-T1",variable=self.measurement_choice,value="CV-T1",command=self.change_method)
        cv_t1_method_button.grid(row=2,column=0,sticky="w")
        swv_t1_method_button = tk.Radiobutton(self.method_choice_frame,text="SWV-T1",variable=self.measurement_choice,value="SWV-T1",command=self.change_method)
        swv_t1_method_button.grid(row=2,column=1,sticky="w")
        lsv_t1_method_button = tk.Radiobutton(self.method_choice_frame,text="LSV-T1",variable=self.measurement_choice,value="LSV-T1",command=self.change_method)
        lsv_t1_method_button.grid(row=2,column=2,sticky="w")
        #Laser spot buttons
        self.laser_spot = tk.IntVar(value=270)
        
        #ESR
        esr_labels = [
            "Dynamic points:",
            "Averages:",
            "Min MW Freq [Ghz]:",
            "MAX MW Freq [Ghz]:",
            f"MW Ampltude [dB] (max. {max_db}):",
            "Annotation:"
        ]

        esr_defaults = [
            self.esr.points,
            self.esr.averages,
            self.esr.min_mw_freq,
            self.esr.max_mw_freq,
            self.esr.mw_amplitude,
           # self.esr.laser_spot_position,
            ""
        ]

        self.esr_entries = []
        esr_frame = tk.Frame(self.measurement_frame)
        self.measurement_methods["ESR"] = esr_frame
        esr_frame.pack()
        
        
        esr_title = tk.Label(esr_frame, text="ESR measurement",  font=("Arial", 12, "bold"))
        esr_title.grid(row=0, column=0, columnspan=2, pady=10)
        esr_spot_frame = tk.Frame(esr_frame)
        esr_spot_frame.grid(row=1,column=0,columnspan=2)
        esr_spot_title = tk.Label(esr_spot_frame, text="Laser position [Mhz]:")
        esr_spot_title.grid(row=0,column=0,columnspan=3)
        
        esr_twothirty_button = tk.Radiobutton(esr_spot_frame, text="230",variable=self.laser_spot,value=230)
        esr_twothirty_button.grid(row=1,column=0)
        esr_twotfifty_button = tk.Radiobutton(esr_spot_frame, text="250",variable=self.laser_spot,value=250)
        esr_twotfifty_button.grid(row=1,column=1)
        esr_twoseventy_button = tk.Radiobutton(esr_spot_frame, text="270",variable=self.laser_spot,value=270)
        esr_twoseventy_button.grid(row=1,column=2)
        
        for i, (label_text, default) in enumerate(zip(esr_labels, esr_defaults)):
            label = tk.Label(esr_frame, text=label_text)
            label.grid(row=i + 2, column=0, sticky="w", padx=5, pady=5)

            entry = tk.Entry(esr_frame)
            entry.insert(0, default)
            entry.grid(row=i + 2, column=1, padx=5, pady=5)

            self.esr_entries.append(entry)

        esr_button = tk.Button(esr_frame, text="Start ESR Experiment", command=self.on_esr_send)
        esr_button.grid(row=len(esr_labels) + 2, column=0, columnspan=2, pady=10)
        
        #RABI     
        rabi_labels = [
            "max. \u03C0-pulse Length [ns]:",
            "Averages:",
            "MW Frequency [Ghz]:",
            f"MW Ampltude [dB] (max. {max_db}):",
            "Annotation:"
            ]
        rabi_defaults = [
            self.rabi.max_pulse_duration,
            self.rabi.averages,
            self.rabi.mw_frequency,
            self.rabi.mw_amplitude,
            ""
            ]
        self.rabi_entries = []
        
        
        rabi_frame = tk.Frame(self.measurement_frame)
        self.measurement_methods["Rabi"] = rabi_frame
        rabi_title = tk.Label(rabi_frame,text="Rabi measurement",font=("Arial", 12, "bold"))
        rabi_title.grid(row=0, column=0, columnspan=2, pady=10)
        
        rabi_spot_frame = tk.Frame(rabi_frame)
        rabi_spot_frame.grid(row=1,column=0,columnspan=2)
        rabi_spot_title = tk.Label(rabi_spot_frame, text="Laser position [Mhz]:")
        rabi_spot_title.grid(row=0,column=0,columnspan=3)
        
        rabi_twothirty_button = tk.Radiobutton(rabi_spot_frame, text="230",variable=self.laser_spot,value=230)
        rabi_twothirty_button.grid(row=1,column=0)
        rabi_twotfifty_button = tk.Radiobutton(rabi_spot_frame, text="250",variable=self.laser_spot,value=250)
        rabi_twotfifty_button.grid(row=1,column=1)
        rabi_twoseventy_button = tk.Radiobutton(rabi_spot_frame, text="270",variable=self.laser_spot,value=270)
        rabi_twoseventy_button.grid(row=1,column=2)
        
        for i, (label_text,default) in enumerate(zip(rabi_labels, rabi_defaults)):
            label = tk.Label(rabi_frame, text=label_text)
            label.grid(row=i + 2, column=0, sticky="w", padx=5, pady=5)
            entry = tk.Entry(rabi_frame)
            entry.insert(0, default)
            entry.grid(row=i + 2, column=1, padx=5, pady=5)
            self.rabi_entries.append(entry)
        rabi_button = tk.Button(rabi_frame, text="Start Rabi Experiment", command=self.on_rabi_send)
        rabi_button.grid(row=len(rabi_labels) + 2, column=0, columnspan=2, pady=10)
        
        #T1
        t1_labels = [
            "Points:",
            "Spacing:",
            "Averages:",
            "Max. \u03C4 [ms]:",
            "\u03C0-pulse Length [ns]:",
            "MW Frequency [Ghz]:",
            f"MW Ampltude [dB] (max. {max_db}):",
            "Annotation:"
            ]
        t1_defaults = [
            self.t1.points,
            self.t1.spacing,
            self.t1.averages,
            self.t1.tau_max,
            self.t1.pi_pulse_length,
            self.t1.mw_frequency,
            self.t1.mw_amplitude,
            ""
            ]
        self.t1_entries = []
        t1_frame = tk.Frame(self.measurement_frame)
        self.measurement_methods["T1"] = t1_frame
        t1_title = tk.Label(t1_frame,text="T1 measurement",font=("Arial", 12, "bold"))
        t1_title.grid(row=0, column=0, columnspan=20, pady=10)
        
        t1_spot_frame = tk.Frame(t1_frame)
        t1_spot_frame.grid(row=1,column=0,columnspan=2)
        t1_spot_title = tk.Label(t1_spot_frame, text="Laser position [Mhz]:")
        t1_spot_title.grid(row=0,column=0,columnspan=3)
        
        t1_twothirty_button = tk.Radiobutton(t1_spot_frame, text="230",variable=self.laser_spot,value=230)
        t1_twothirty_button.grid(row=1,column=0)
        t1_twotfifty_button = tk.Radiobutton(t1_spot_frame, text="250",variable=self.laser_spot,value=250)
        t1_twotfifty_button.grid(row=1,column=1)
        t1_twoseventy_button = tk.Radiobutton(t1_spot_frame, text="270",variable=self.laser_spot,value=270)
        t1_twoseventy_button.grid(row=1,column=2)
        
        for i, (label_text,default) in enumerate(zip(t1_labels, t1_defaults)):
            label = tk.Label(t1_frame, text=label_text)
            label.grid(row=i + 2, column=0, sticky="w", padx=5, pady=5)
            entry = tk.Entry(t1_frame)
            entry.insert(0, default)
            entry.grid(row=i + 2, column=1, padx=5, pady=5)
            self.t1_entries.append(entry)
        t1_button = tk.Button(t1_frame, text="Start T1 Experiment", command=self.on_t1_send)
        t1_button.grid(row=len(t1_labels) + 2, column=0, columnspan=2, pady=10)
        
        
        
        #CV
        cv_labels = ["Start Potential [V]:","End Potential [V]:","Potential Step [mV]:","Sweeping Rate [mV/s]:","Number of Measurements:","Annotation:"]
        cv_defaults= [0,1,10,50,1,""]
        self.cv_entries = []
        cv_frame = tk.Frame(self.measurement_frame)
        self.measurement_methods["CV"] = cv_frame
        cv_title = tk.Label(cv_frame,text="CV Measurement",font=("Arial", 12, "bold"))
        cv_title.grid(row=0, column=0, columnspan=20, pady=10)
        for i, (label_text,default) in enumerate(zip(cv_labels, cv_defaults)):
            label = tk.Label(cv_frame, text=label_text)
            label.grid(row=i + 1, column=0, sticky="w", padx=5, pady=5)
            entry = tk.Entry(cv_frame)
            entry.insert(0, default)
            entry.grid(row=i + 1, column=1, padx=5, pady=5)
            self.cv_entries.append(entry)
        cv_button = tk.Button(cv_frame, text="Start CV Experiment", command=self.on_cv_send)
        cv_button.grid(row=len(cv_labels) + 1, column=0, columnspan=2, pady=10)
        
        #SWV
        swv_labels = ["Start Potential [V]:","End Potential [V]:","Potential Step [mV]:","Frequency [Hz]:","Amplitude [mV]","Number of Measurements","Annotation:"]
        swv_defaults= [0,1,10,20,100,1,""]
        self.swv_entries = []
        swv_frame = tk.Frame(self.measurement_frame)
        self.measurement_methods["SWV"] = swv_frame
        swv_title = tk.Label(swv_frame,text="SWV Measurement",font=("Arial", 12, "bold"))
        swv_title.grid(row=0, column=0, columnspan=20, pady=10)
        for i, (label_text,default) in enumerate(zip(swv_labels, swv_defaults)):
            label = tk.Label(swv_frame, text=label_text)
            label.grid(row=i + 1, column=0, sticky="w", padx=5, pady=5)
            entry = tk.Entry(swv_frame)
            entry.insert(0, default)
            entry.grid(row=i + 1, column=1, padx=5, pady=5)
            self.swv_entries.append(entry)
        swv_button = tk.Button(swv_frame, text="Start SWV Experiment", command=self.on_swv_send)
        swv_button.grid(row=len(swv_labels) + 1, column=0, columnspan=2, pady=10)
        
        #LSV
        lsv_labels = ["Start Potential [V]:","End Potential [V]:","Potential Step [mV]:","Scanrate [mV/s]:","Number of Measurements","Annotation:"]
        lsv_defaults= [0,1,10,50,1,""]
        self.lsv_entries=[]
        lsv_frame = tk.Frame(self.measurement_frame)
        self.measurement_methods["LSV"] = lsv_frame
        lsv_title = tk.Label(lsv_frame,text="LSV Measurement",font=("Arial", 12, "bold"))
        lsv_title.grid(row=0, column=0, columnspan=20, pady=10)
        for i, (label_text, default) in enumerate(zip(lsv_labels,lsv_defaults)):
            label = tk.Label(lsv_frame,text=label_text)
            label.grid(row=i + 1, column=0, sticky="w", padx=5, pady=5)
            entry = tk.Entry(lsv_frame)
            entry.insert(0,default)
            entry.grid(row=i + 1, column=1, padx=5, pady=5)
            self.lsv_entries.append(entry)
        lsv_button = tk.Button(lsv_frame,text="Start LCV Experiment",command=self.on_lsv_send)
        lsv_button.grid(row=len(lsv_labels) + 1, column=0, columnspan=2, pady=10)
        
        #CV-T1
        cv_t1_labels = [
            "Points:",
            "Spacing:",
            "Max. \u03C4 [ms]:",
            "\u03C0-pulse Length [ns]:",
            "MW Frequency [Ghz]:",
            f"MW Ampltude [dB] (max. {max_db}):",
            "Start Potential [V]:",
            "End Potential [V]:",
            "Potential Step [mV]:",
            "Sweeping Rate [mV/s]:",
            "T1-averages / CV-point:",
            "Annotation:"
            ]
        cv_t1_defaults = [
            self.t1.points,
            self.t1.spacing,
            self.t1.tau_max,
            self.t1.pi_pulse_length,
            self.t1.mw_frequency,
            self.t1.mw_amplitude,
            0,
            1,
            10,
            50,
            300,
            ""
            ]
        self.cv_t1_entries = []
        cv_t1_frame = tk.Frame(self.measurement_frame)
        self.measurement_methods["CV-T1"] = cv_t1_frame
        cv_t1_title = tk.Label(cv_t1_frame,text="CV-T1 measurement",font=("Arial", 12, "bold"))
        cv_t1_title.grid(row=0, column=0, columnspan=20, pady=10)
        
        cv_t1_spot_frame = tk.Frame(cv_t1_frame)
        cv_t1_spot_frame.grid(row=1,column=0,columnspan=4)
        cv_t1_spot_title = tk.Label(cv_t1_spot_frame, text="Laser position [Mhz]:")
        cv_t1_spot_title.grid(row=0,column=0,columnspan=3)
        
        cv_t1_twothirty_button = tk.Radiobutton(cv_t1_spot_frame, text="230",variable=self.laser_spot,value=230)
        cv_t1_twothirty_button.grid(row=1,column=0)
        cv_t1_twotfifty_button = tk.Radiobutton(cv_t1_spot_frame, text="250",variable=self.laser_spot,value=250)
        cv_t1_twotfifty_button.grid(row=1,column=1)
        cv_t1_twoseventy_button = tk.Radiobutton(cv_t1_spot_frame, text="270",variable=self.laser_spot,value=270)
        cv_t1_twoseventy_button.grid(row=1,column=2)
        
        
        for i, (label_text,default) in enumerate(zip(cv_t1_labels, cv_t1_defaults)):
            label = tk.Label(cv_t1_frame, text=label_text)
            entry = tk.Entry(cv_t1_frame)
            entry.insert(0, default)
            if i%2 ==0: 
                label.grid(row=i + 2, column=0, sticky="w", padx=5, pady=5)
                entry.grid(row=i + 2, column=1, padx=5, pady=5)
            else:
                label.grid(row=i + 1, column=2, sticky="w", padx=5, pady=5)
                entry.grid(row=i + 1, column=3, padx=5, pady=5)
            self.cv_t1_entries.append(entry)
        cv_t1_button = tk.Button(cv_t1_frame, text="Start CV-T1 Experiment", command=self.on_cv_t1_send)
        cv_t1_button.grid(row=len(cv_t1_labels) + 2, column=0, columnspan=4, pady=10)
        
        #SWV-T1
        swv_t1_labels = [
            "Points:",
            "Spacing:",
            "Max. \u03C4 [ms]:",
            "\u03C0-pulse Length [ns]:",
            "MW Frequency [Ghz]:",
            f"MW Ampltude [dB] (max. {max_db}):",
            "Start Potential [V]:",
            "End Potential [V]:",
            "Potential Step [mV]:",
            "Frequency [Hz]:",
            "Amplitude [mV]",
            "T1-averages / SWV-point:",
            "Annotation:"
            ]
        swv_t1_defaults = [
            self.t1.points,
            self.t1.spacing,
            self.t1.tau_max,
            self.t1.pi_pulse_length,
            self.t1.mw_frequency,
            self.t1.mw_amplitude,
            0,
            1,
            10,
            20,
            100,
            300,
            ""
            ]
        self.swv_t1_entries = []
        swv_t1_frame = tk.Frame(self.measurement_frame)
        self.measurement_methods["SWV-T1"] = swv_t1_frame
        swv_t1_title = tk.Label(swv_t1_frame,text="SWV-T1 measurement",font=("Arial", 12, "bold"))
        swv_t1_title.grid(row=0, column=0, columnspan=20, pady=10)
        
        swv_t1_spot_frame = tk.Frame(swv_t1_frame)
        swv_t1_spot_frame.grid(row=1,column=0,columnspan=4)
        swv_t1_spot_title = tk.Label(swv_t1_spot_frame, text="Laser position [Mhz]:")
        swv_t1_spot_title.grid(row=0,column=0,columnspan=3)
        
        swv_t1_twothirty_button = tk.Radiobutton(swv_t1_spot_frame, text="230",variable=self.laser_spot,value=230)
        swv_t1_twothirty_button.grid(row=1,column=0)
        swv_t1_twotfifty_button = tk.Radiobutton(swv_t1_spot_frame, text="250",variable=self.laser_spot,value=250)
        swv_t1_twotfifty_button.grid(row=1,column=1)
        swv_t1_twoseventy_button = tk.Radiobutton(swv_t1_spot_frame, text="270",variable=self.laser_spot,value=270)
        swv_t1_twoseventy_button.grid(row=1,column=2)
        
        
        for i, (label_text,default) in enumerate(zip(swv_t1_labels, swv_t1_defaults)):
            label = tk.Label(swv_t1_frame, text=label_text)
            entry = tk.Entry(swv_t1_frame)
            entry.insert(0, default)
            if i%2 ==0: 
                label.grid(row=i + 2, column=0, sticky="w", padx=5, pady=5)
                entry.grid(row=i + 2, column=1, padx=5, pady=5)
            else:
                label.grid(row=i + 1, column=2, sticky="w", padx=5, pady=5)
                entry.grid(row=i + 1, column=3, padx=5, pady=5)
            self.swv_t1_entries.append(entry)
        swv_t1_button = tk.Button(swv_t1_frame, text="Start SWV-T1 Experiment", command=self.on_swv_t1_send)
        swv_t1_button.grid(row=len(swv_t1_labels) + 2, column=0, columnspan=4, pady=10)
       
        #LSV-T1
        lsv_t1_labels = [ "Points:",
                         "Spacing:",
         "Max. \u03C4 [ms]:",
         "\u03C0-pulse Length [ns]:",
         "MW Frequency [Ghz]:",
         f"MW Ampltude [dB] (max. {max_db}):",
         "Start Potential [V]:","End Potential [V]:","Potential Step [mV]:","Scanrate [mV/s]:","T1-averages / LSV-point:","Annotation:"]
        lsv_t1_defaults= [
            self.t1.points,
            self.t1.spacing,
            self.t1.tau_max,
            self.t1.pi_pulse_length,
            self.t1.mw_frequency,
            self.t1.mw_amplitude,
            0,
            1,
            10,
            50,
            300,
            ""]
        
        self.lsv_t1_entries = []
        lsv_t1_frame = tk.Frame(self.measurement_frame)
        self.measurement_methods["LSV-T1"] = lsv_t1_frame
        lsv_t1_title = tk.Label(lsv_t1_frame,text="LSV-T1 measurement",font=("Arial", 12, "bold"))
        lsv_t1_title.grid(row=0, column=0, columnspan=20, pady=10)
        
        lsv_t1_spot_frame = tk.Frame(lsv_t1_frame)
        lsv_t1_spot_frame.grid(row=1,column=0,columnspan=4)
        lsv_t1_spot_title = tk.Label(lsv_t1_spot_frame, text="Laser position [Mhz]:")
        lsv_t1_spot_title.grid(row=0,column=0,columnspan=3)
        
        lsv_t1_twothirty_button = tk.Radiobutton(lsv_t1_spot_frame, text="230",variable=self.laser_spot,value=230)
        lsv_t1_twothirty_button.grid(row=1,column=0)
        lsv_t1_twotfifty_button = tk.Radiobutton(lsv_t1_spot_frame, text="250",variable=self.laser_spot,value=250)
        lsv_t1_twotfifty_button.grid(row=1,column=1)
        lsv_t1_twoseventy_button = tk.Radiobutton(lsv_t1_spot_frame, text="270",variable=self.laser_spot,value=270)
        lsv_t1_twoseventy_button.grid(row=1,column=2)
        
        
        for i, (label_text,default) in enumerate(zip(lsv_t1_labels, lsv_t1_defaults)):
            label = tk.Label(lsv_t1_frame, text=label_text)
            entry = tk.Entry(lsv_t1_frame)
            entry.insert(0, default)
            if i%2 ==0: 
                label.grid(row=i + 2, column=0, sticky="w", padx=5, pady=5)
                entry.grid(row=i + 2, column=1, padx=5, pady=5)
            else:
                label.grid(row=i + 1, column=2, sticky="w", padx=5, pady=5)
                entry.grid(row=i + 1, column=3, padx=5, pady=5)
            self.lsv_t1_entries.append(entry)
        lsv_t1_button = tk.Button(lsv_t1_frame, text="Start LSV-T1 Experiment", command=self.on_lsv_t1_send)
        lsv_t1_button.grid(row=len(lsv_t1_labels) + 2, column=0, columnspan=4, pady=10)
        
        #Treeview loader
        self.load = tk.Frame(measure)
        self.load.grid(row=0,column=1,sticky="nsew", padx=5, pady=5)
        loader_title = tk.Label(self.load, text="Load file from current Folder")
        loader_title.pack()
        self.loader = ttk.Treeview(self.load, columns=("type", "method", "date","time", "annotation"),
                                   show="headings")
        self.loader.heading("type", text="Type")
        self.loader.heading("method", text="Method")
        self.loader.heading("date", text="Date")
        self.loader.heading("time", text="Time")
        self.loader.heading("annotation", text="Annotation")
        self.loader.bind("<Double-1>", self.file_selected)
        self.loader.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        #live
        
        live = tk.Frame(scrollable_frame)
        self.frames["live"] = live
        live.columnconfigure(0, weight=2)  # LEFT (2/3)
        live.columnconfigure(1, weight=1)  # RIGHT (1/3)
        live.rowconfigure(0, weight=1)
        live.rowconfigure(1, weight=1)
        live.rowconfigure(2, weight=1)

        live_info_frame = tk.Frame(live)
        live_info_frame.grid(row=2,column=1,sticky="nsew", padx=5, pady=5)
        live_info_frame.grid_propagate(True)
        self.live_info_text = tk.Text(live_info_frame,height=10)
        self.live_info_text.pack(padx=5,pady=10,expand=False)
        cancel_button = tk.Button(live_info_frame,text="Cancel measurement", command=self.cancel_measurement)
        cancel_button.pack()
        self.live_status_label = tk.Label(live, text="")
        self.live_status_label.grid(row=0,column=0,columnspan=2,padx=5,pady=5)
        self.live_progress = ttk.Progressbar(live, length=300, mode="determinate")
        self.live_progress.grid(row=1,column=0,columnspan=2,padx=5,pady=5)
        self.live_plot_container = tk.Frame(live, width=600, height=400)
        self.live_plot_container.grid(row=2,column=0,sticky="nsew", padx=5, pady=5)
        self.live_canvas = None
        
        #eval
        
        evaluator = tk.Frame(scrollable_frame)
        self.frames["evaluator"] = evaluator
        evaluator.columnconfigure(0, weight=2)  # LEFT (2/3)
        evaluator.columnconfigure(1, weight=1)  # RIGHT (1/3)
        evaluator.rowconfigure(0, weight=1)
        self.final_canvas = None
        self.final_plot_container = tk.Frame(evaluator, width=600, height=400)
        self.final_plot_container.grid(row=0,column=0,sticky="nsew", padx=5, pady=5)
        
        self.fitting_params_frame = tk.Frame(evaluator, bd=2, relief="groove")
        self.fitting_params_frame.grid(row=0, column=1, sticky="nsew",padx=5,pady=5)
        self.fitting_params_titles = tk.Label(self.fitting_params_frame, text="Data evaluation", font=("Arial", 12, "bold"))
        self.fitting_params_titles.pack()
        
        #ESR Fitting Params
        
        esr_fitting_frame = tk.Frame(self.fitting_params_frame)
        self.fitting_frames["ESR"] = esr_fitting_frame
        esr_fitting_title = tk.Label(esr_fitting_frame, text="ESR peak finding")
        esr_fitting_title.pack()

        esr_fit_param_frame = tk.Frame(esr_fitting_frame)
        esr_fit_param_frame.pack()
        esr_fitting_labels = ["Fitting:","Prominence:","Distance:"]
        esr_fitting_defaults = [True, 0.01, 5]
        
        self.esr_fitting_entries = self.layout_fitting_params(esr_fit_param_frame, esr_fitting_labels, esr_fitting_defaults)
        self.esr_fitting_button = tk.Button(esr_fitting_frame,text="Fit Data",command=self.esr_fitting)
        self.esr_fitting_button.pack()

            
        
        
        #T1 Fitting Params
        t1_fitting_frame = tk.Frame(self.fitting_params_frame)
        self.fitting_frames["T1"] = t1_fitting_frame
        t1_fitting_title = tk.Label(t1_fitting_frame, text="T1 exp. curve fitting")
        t1_fitting_title.pack()
        
        t1_fit_param_frame = tk.Frame(t1_fitting_frame)
        t1_fit_param_frame.pack()
        t1_fitting_labels = ["Fitting:","Use stretched exponential fit:","Stretched exp. fit - Max. tries:","Stretched exp. fit - use Bounds:","Show stretched exp. fit:",
                             "Use Biexpontential fit:","biexp. fit - Max. tries:","biexp.fit - use Bounds:","Show biexp. fit:"]
        t1_fitting_defaults = [True, True, 20000, True, True, False, 20000, True, True]
        self.t1_fitting_entries = self.layout_fitting_params(t1_fit_param_frame, t1_fitting_labels, t1_fitting_defaults)
        self.t1_fitting_button = tk.Button(t1_fitting_frame, text="Fit Data", command=self.t1_fitting)
        self.t1_fitting_button.pack()
        #Rabi Fitting Params
        rabi_fitting_frame = tk.Frame(self.fitting_params_frame)
        self.fitting_frames["Rabi"] = rabi_fitting_frame
        rabi_fitting_title = tk.Label(rabi_fitting_frame, text="Rabi sinus curve fitting")
        rabi_fitting_title.pack()
        
        rabi_fitting_labels = ["Fitting:","Use Sinus_fit:","Show Sinus fit:","Use Bounds:","Max. Tries:","Use Peakfinding:","Prominence:","Distance:"]
        rabi_fitting_defaults = [True, True, True, True, 20000, False, 0.01, 20]
        rabi_fitting_param_frame = tk.Frame(rabi_fitting_frame)
        rabi_fitting_param_frame.pack()
        self.rabi_fitting_entries = self.layout_fitting_params(rabi_fitting_param_frame, rabi_fitting_labels, rabi_fitting_defaults)    
        self.rabi_fitting_button = tk.Button(rabi_fitting_frame, text="Fit Data", command=self.rabi_fitting)
        self.rabi_fitting_button.pack()
        #CV Fitting params
        cv_fitting_frame = tk.Frame(self.fitting_params_frame)
        self.fitting_frames["CV"] = cv_fitting_frame
        cv_fitting_title = tk.Label(cv_fitting_frame, text="CV peak finding")
        cv_fitting_title.pack()

        cv_fit_param_frame = tk.Frame(cv_fitting_frame)
        cv_fit_param_frame.pack()
        cv_fitting_labels = ["Fitting:","Prominence:","Distance:"]
        cv_fitting_defaults = [True, 1e-6, 10]
        
        self.cv_fitting_entries = self.layout_fitting_params(cv_fit_param_frame, cv_fitting_labels, cv_fitting_defaults)
        self.cv_fitting_button = tk.Button(cv_fitting_frame,text="Fit Data",command=self.cv_fitting)
        self.cv_fitting_button.pack()
        #SWV Fitting params
        swv_fitting_frame = tk.Frame(self.fitting_params_frame)
        self.fitting_frames["SWV"] = swv_fitting_frame
        swv_fitting_title = tk.Label(swv_fitting_frame, text="SWV peak finding")
        swv_fitting_title.pack()

        swv_fit_param_frame = tk.Frame(swv_fitting_frame)
        swv_fit_param_frame.pack()
        swv_fitting_labels = ["Fitting:","Prominence:","Distance:"]
        swv_fitting_defaults = [True, 1e-6, 10]
        
        self.swv_fitting_entries = self.layout_fitting_params(swv_fit_param_frame, swv_fitting_labels, swv_fitting_defaults)
        self.swv_fitting_button = tk.Button(swv_fitting_frame,text="Fit Data",command=self.swv_fitting)
        self.swv_fitting_button.pack()
        #LSV-fitting
        lsv_fitting_frame = tk.Frame(self.fitting_params_frame)
        self.fitting_frames["LSV"] = lsv_fitting_frame
        lsv_fitting_title = tk.Label(lsv_fitting_frame, text="LSV peak finding")
        lsv_fitting_title.pack()
        lsv_fit_param_frame = tk.Frame(lsv_fitting_frame)
        lsv_fit_param_frame.pack()
        lsv_fitting_labels = ["Fitting:","Prominence:","Distance:"]
        lsv_fitting_defaults = [True, 1e-6, 10]
        
        self.lsv_fitting_entries = self.layout_fitting_params(lsv_fit_param_frame, lsv_fitting_labels, lsv_fitting_defaults)
        self.lsv_fitting_button = tk.Button(lsv_fitting_frame,text="Fit Data",command=self.lsv_fitting)
        self.lsv_fitting_button.pack()
        
        ECQS_labels = ["Fitting:","Prominence:","Distance:","Cutoff Rate","Use Bounds:","Max. Tries:"]
        ECQS_standards = [True, 1e-6,10,0.5,False,5000]
        
        #CV-T1 Fitting
        cv_t1_fitting_frame = tk.Frame(self.fitting_params_frame)
        self.fitting_frames["CV-T1"] = cv_t1_fitting_frame
        cv_t1_fitting_title = tk.Label(cv_t1_fitting_frame, text="CV-T1 fitting")
        cv_t1_fitting_title.pack()
        
        cv_t1_fit_param_frame = tk.Frame(cv_t1_fitting_frame)
        cv_t1_fit_param_frame.pack()
        cv_t1_fitting_labels = ECQS_labels
        cv_t1_fitting_defaults = ECQS_standards
        self.cv_t1_fitting_entries = self.layout_fitting_params(cv_t1_fit_param_frame, cv_t1_fitting_labels, cv_t1_fitting_defaults)
        self.cv_t1_fitting_button = tk.Button(cv_t1_fitting_frame, text="Fit Data",command=self.cv_t1_fitting)
        self.cv_t1_fitting_button.pack()
        #SWV-T1 Fitting
        swv_t1_fitting_frame = tk.Frame(self.fitting_params_frame)
        self.fitting_frames["SWV-T1"] = swv_t1_fitting_frame
        swv_t1_fitting_title = tk.Label(swv_t1_fitting_frame, text="SWV-T1 fitting")
        swv_t1_fitting_title.pack()     
        
        swv_t1_fit_param_frame = tk.Frame(swv_t1_fitting_frame)
        swv_t1_fit_param_frame.pack()
        swv_t1_fitting_labels = ECQS_labels
        swv_t1_fitting_defaults = ECQS_standards
        self.swv_t1_fitting_entries = self.layout_fitting_params(swv_t1_fit_param_frame, swv_t1_fitting_labels, swv_t1_fitting_defaults)
        self.swv_t1_fitting_button = tk.Button(swv_t1_fitting_frame, text="Fit Data",command=self.swv_t1_fitting)
        self.swv_t1_fitting_button.pack()
        #LSV-T1
        lsv_t1_fitting_frame = tk.Frame(self.fitting_params_frame)
        self.fitting_frames["LSV-T1"] = lsv_t1_fitting_frame
        lsv_t1_fitting_title = tk.Label(lsv_t1_fitting_frame, text="LSV-T1 fitting")
        lsv_t1_fitting_title.pack()     
        
        lsv_t1_fit_param_frame = tk.Frame(lsv_t1_fitting_frame)
        lsv_t1_fit_param_frame.pack()
        lsv_t1_fitting_labels = ECQS_labels
        lsv_t1_fitting_defaults = ECQS_standards
        self.lsv_t1_fitting_entries = self.layout_fitting_params(lsv_t1_fit_param_frame, lsv_t1_fitting_labels, lsv_t1_fitting_defaults)
        self.lsv_t1_fitting_button = tk.Button(lsv_t1_fitting_frame, text="Fit Data",command=self.lsv_t1_fitting)
        self.lsv_t1_fitting_button.pack()
        
        
        self.save_figure_button = tk.Button(self.fitting_params_frame,text="Save Figure",command=self.save_figure)
        self.save_figure_button.pack()
        self.eval_info_text = tk.Text(self.fitting_params_frame,height=10)
        self.eval_info_text.pack(padx=5,pady=10,expand=False)
        
        
        self.focus_force()
    def change_method(self):
        for old_frame in self.measurement_methods.values():
            old_frame.pack_forget()
        self.measurement_methods[str(self.measurement_choice.get())].pack()
    def show_frame(self, frame: str | list) -> None:
        for old_frame in self.frames.values():
            old_frame.pack_forget()
        if isinstance(frame, list):
            for single_frame in frame:
                self.frames[single_frame].pack(padx=20, pady=20, fill="both", expand=True)
        else:
            self.frames[frame].pack(padx=20, pady=20, fill="both", expand=True)
    def fitting_parameter_update(self):
        for old_fit in self.fitting_frames.values():
            old_fit.pack_forget()
        self.fitting_frames[self.current_measurement_method].pack(padx=20, pady=20, fill="both", expand=True)
    def layout_fitting_params(self,frame, parameter_list: list, default_list: list) -> list:
        entries = []
        for i, (label_text, default) in enumerate(zip(parameter_list, default_list)):
            label = tk.Label(frame, text=label_text)
            label.grid(row=i, column=0,padx=5, pady=5)
            if isinstance(default, bool):
                var = tk.BooleanVar(value=default)
                checkbox=tk.Checkbutton(frame, variable=var)
                checkbox.grid(row=i,column=1, padx=5,pady=5)
                entries.append(("bool",var))
            else:
                entry = tk.Entry(frame)
                entry.insert(0,default)
                entry.grid(row=i,column=1, padx=5,pady=5)
                entries.append(("entry",entry))
        return entries
                
    def get_fitting_value(self, item):
        typ, obj = item
        value = obj.get()
        if typ == "entry":
            # try to convert to number if possible
            try:
                if "." in value or "e" in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value
        return value            
    def measurement_info_update(self, text:dict = {}) -> None:
        self.eval_info_text.delete("1.0", tk.END)
        self.live_info_text.delete("1.0", tk.END)
        for key,val in text.items():
            self.eval_info_text.insert(tk.END, f"{key}: {val}\n") 
            self.live_info_text.insert(tk.END, f"{key}: {val}\n") 
    def show_boot_frame(self,booting: bool) -> None:
        if booting:
            self.measurement_frame.grid_forget()
            self.boot.grid(row=0, column=0)
        else:
            self.boot.grid_forget()
            self.measurement_frame.grid(row=0,column=0)
    def on_resize(self,event):
        # Only react to the root window resizing
        if event.widget == self:
            #print("New size:", event.width, event.height)
            pass
    def cancel_live_update_job(self):
        if getattr(self, "live_update_job", None) is not None:
            try:
                self.after_cancel(self.live_update_job)
            except Exception:
                pass
            self.live_update_job = None
    def reset_final_canvas(self):
        if getattr(self, "final_canvas", None) is not None:
            old_fig = self.final_canvas.figure
            self.final_canvas.get_tk_widget().destroy()
            self.final_canvas = None
            if old_fig is not None:
                plt.close(old_fig)
    def cleanup_measurement_state(self):
        self.cancel_live_update_job()
        self.cancel_requested = False
        if getattr(self, "qs_data_reader", None) is not None:
            try:
                self.qs_data_reader.release_resources()
            except Exception:
                pass
            self.qs_data_reader = None
        if getattr(self, "potentiostat", None) is not None:
            try:
                self.potentiostat.release_resources()
            except Exception:
                pass
            self.potentiostat = None
        if getattr(self, "ecqs_cache", None) is not None:
            try:
                cache_fig = self.ecqs_cache.fig
                self.ecqs_cache.clear()
                if cache_fig is not None and cache_fig is not getattr(self, "fig", None):
                    plt.close(cache_fig)
            except Exception:
                pass
        self.reset_live_canvas()
        self.reset_final_canvas()
        self.fig = None
        gc.collect()
    def on_close(self):
        try: 
            self.cleanup_measurement_state()
            self.qupyt.exit()          
        except:
            print("Qupyt never started")
        finally:
            
            self.destroy()
            
    def update_file_loader(self):
        self.loader.delete(*self.loader.get_children())
        measurements = parse_file_folder(self.data_folder)     
        measurements.sort(key=lambda x: (x["date"], x["time"]),reverse=True)
        for m in measurements:
            self.loader.insert(
                "", "end",iid=m["file_name"], values=(
                    m["type"],
                    m["method"],
                    m["date"],
                    m["time"],
                    m["annotation"]
                ))
    def file_selected(self, event):
        item_id = self.loader.selection()
        if not item_id:
            return
        iid = item_id[0]
        values = self.loader.item(iid, "values")
        if values[0] == "QS":
            self.cleanup_measurement_state()
            self.qs_file_name = iid
            self.current_measurement_method = values[1]
            self.qs_data_reader = QS_data_reader(self.data_folder, iid)
            self.qs_data_reader.start_data_building()
            self.show_frame("live")
            self.update_qs_live_figure()
        if values[0] == "EC":
            self.cleanup_measurement_state()
            self.potentiostat = Potentiostat(self.data_folder)
            self.ec_file_name = iid
            self.current_measurement_method=values[1]
            self.potentiostat.load_measurement_from_file(iid)
            self.show_frame("live")
            self.update_ec_live_figure()
        if values[0] == "ECQS":
            self.cleanup_measurement_state()
            self.potentiostat = Potentiostat(self.data_folder)
            ecqs_files = []
            used_methods = []
            for all_item_id in self.loader.get_children():
                all_values = self.loader.item(all_item_id,"values")
                if all_values[2] == values[2] and all_values[3] == values[3]:
                    ecqs_files.append(all_item_id)
                    used_methods.append(all_values[1])
            if "CV" in used_methods and "T1" in used_methods:
                self.current_measurement_method = "CV-T1"
            elif "SWV" in used_methods and "T1" in used_methods:
                self.current_measurement_method = "SWV-T1"
            elif "LSV" in used_methods and "T1" in used_methods:
                self.current_measurement_method = "LSV-T1"
            for entry in ecqs_files:
                splitted = entry.split("_")
                if splitted[1] == "CV" or splitted[1] == "SWV" or splitted[1] == "LSV":
                    self.ec_file_name = entry
                elif splitted[1] == "T1":
                    self.qs_file_name=entry
            self.qs_data_reader = QS_data_reader(self.data_folder, self.qs_file_name)
            self.qs_data_reader.start_data_building()
            self.potentiostat.load_measurement_from_file(self.ec_file_name)
            self.show_frame("live")
            self.live_update_job = self.after(200, self.update_ecqs_live_figure)
          

    def on_folder_send(self):
        self.data_folder = filedialog.askdirectory()
        if os.path.isdir(self.data_folder):
            self.show_boot_frame(True)
            self.boot_progress.start(10)
            self.update_file_loader()
            self.show_frame("measure")
            self.boot_devices()
            self.focus_force()
            self.after(7000, self.finish_boot)           
    def boot_devices(self):
        self.qupyt = Qupyt_controller(self.data_folder)
        self.potentiostat = Potentiostat(self.data_folder)
        self.qupyt.start()
    def finish_boot(self):
        self.focus_force()
        self.boot_progress.stop()
        #self.show_frame("measure")
        self.show_boot_frame(False)
        self.update_file_loader()
    def finalize_cancelled_measurement(self):
        if not self.cancel_requested:
            return
        self.cancel_requested = False
        if getattr(self, "qupyt", None) is not None:
            try:
                self.qupyt.start()
            except Exception:
                pass
    def cancel_measurement(self):
        self.cancel_requested = True
        if getattr(self, "qs_data_reader", None) is not None:
            try:
                self.qs_data_reader.dataframe_loop_stop.set()
            except Exception:
                pass
        if getattr(self, "potentiostat", None) is not None:
            try:
                self.potentiostat.stop_measurement.set()
            except Exception:
                pass
        if getattr(self, "qupyt", None) is not None:
            try:
                self.qupyt.exit()
            except Exception:
                pass
        self.live_status_label.config(text="Stopping measurement...")
    def on_esr_send(self):
        self.esr = User_ESR_input()
        self.cleanup_measurement_state()
        values = [e.get() for e in self.esr_entries]
        self.esr.points = values[0]
        self.esr.averages = values[1]
        self.esr.min_mw_freq = values[2]
        self.esr.max_mw_freq = values[3]
        self.esr.mw_amplitude = values[4]
        self.esr.laser_spot_position = float(self.laser_spot.get())
        user_params = self.esr.export_params()
        self.qs_file_name, _ = create_file_name(qs_type="ESR", annotation=values[5])
        Send_esr_measurement(self.qs_file_name, user_params)
        self.current_measurement_method = "ESR"
        self.qs_data_reader = QS_data_reader(self.data_folder, self.qs_file_name)
        self.qs_data_reader.start_data_building()
        self.show_frame("live")
        self.update_qs_live_figure()
    def on_rabi_send(self):
        self.rabi = User_Rabi_input()
        self.cleanup_measurement_state()
        values = [e.get() for e in self.rabi_entries]
        self.rabi.max_pulse_duration = values[0]
        self.rabi.averages = values[1]
        self.rabi.mw_frequency = values[2]
        self.rabi.mw_amplitude = values[3]
        self.rabi.laser_spot_position = float(self.laser_spot.get())
        user_params = self.rabi.export_params()
        self.qs_file_name, _ = create_file_name(qs_type="Rabi", annotation=values[4])
        Send_rabi_measurement(self.qs_file_name, user_params)
        self.current_measurement_method = "Rabi"
        self.qs_data_reader = QS_data_reader(self.data_folder, self.qs_file_name)
        self.qs_data_reader.start_data_building()
        self.show_frame("live")
        self.update_qs_live_figure()
    def on_t1_send(self):
        self.t1 = User_T1_input()
        self.cleanup_measurement_state()
        values = [e.get() for e in self.t1_entries]
        self.t1.points = values[0]
        self.t1.spacing = values[1]
        self.t1.averages = values[2]
        self.t1.tau_max = values[3]
        self.t1.pi_pulse_length = values[4]
        self.t1.mw_frequency = values[5]
        self.t1.mw_amplitude = values[6]
        self.t1.laser_spot_position = float(self.laser_spot.get())
        user_params = self.t1.export_params()
        self.qs_file_name, _ = create_file_name(qs_type="T1", annotation=values[7])
        Send_t1_measurement(self.qs_file_name, user_params)
        self.current_measurement_method = "T1"
        self.qs_data_reader = QS_data_reader(self.data_folder, self.qs_file_name)
        self.qs_data_reader.start_data_building()
        self.show_frame("live")
        self.update_qs_live_figure()
    def on_cv_send(self):
        self.cleanup_measurement_state()
        self.potentiostat = Potentiostat(self.data_folder)
        self.ecqs_cache = ECQS_cache()
        # ["Start Potential [V]","End Potential [V]","Potential Step [mV]","Sweeping Rate [mV/s]","Number of Measurements","Annotation"]
        values = [e.get() for e in self.cv_entries]
        method = self.potentiostat.cv_measurement(float(values[0]), float(values[1]), float(values[2])/1000, float(values[3])/1000)
        self.current_measurement_method="CV"
        _,self.ec_file_name = create_file_name(ec_type="CV",annotation=values[5])
        self.potentiostat.start_n_measurements(self.ec_file_name, method, int(values[4]))
        self.show_frame("live")
        self.update_ec_live_figure()
    def on_swv_send(self):
        self.cleanup_measurement_state()
        self.potentiostat = Potentiostat(self.data_folder)
        #["Start Potential [V]:","End Potential [V]:","Potential Step [mV]:","Frequency [Hz]:","Amplitude [mV]","Number of Measurements","Annotation:"]
        values = [e.get() for e in self.swv_entries]
        method = self.potentiostat.swv_measurement(float(values[0]), float(values[1]), float(values[2])/1000, float(values[3]), float(values[4])/1000)
        self.current_measurement_method="SWV"
        _,self.ec_file_name = create_file_name(ec_type="SWV",annotation=values[6])
        self.potentiostat.start_n_measurements(self.ec_file_name, method, int(values[5]))
        self.show_frame("live")
        self.update_ec_live_figure()
        
    def on_lsv_send(self):
        self.cleanup_measurement_state()
        self.potentiostat = Potentiostat(self.data_folder)
        self.ecqs_cache = ECQS_cache()
        #["Start Potential [V]:","End Potential [V]:","Potential Step [mV]:","Scanrate [mV/s]:","Number of Measurements","Annotation:"]
        values = [e.get() for e in self.lsv_entries]
        method = self.potentiostat.lsv_measurement(float(values[0]), float(values[1]), float(values[2])/1000, float(values[3])/1000)
        self.current_measurement_method="LSV"
        _,self.ec_file_name = create_file_name(ec_type="LSV",annotation=values[5])
        self.potentiostat.start_n_measurements(self.ec_file_name, method, int(values[4]))
        self.show_frame("live")
        self.update_ec_live_figure()
            
    def on_cv_t1_send(self):
        self.t1 = User_T1_input()
        self.cleanup_measurement_state()
        self.potentiostat = Potentiostat(self.data_folder)
        self.ecqs_cache = ECQS_cache()
        #  "Points:","Max. \u03C4 [ms]:","\u03C0-pulse Length [ns], MW Frequency [Ghz]:","MW Amplitude [dB]:",
        #"Start Potential [V]:","End Potential [V]:","Potential Step [mV]:","Sweeping Rate [mV/s]:","T1-averages / CV-point:","Annotation:"
        values = [e.get() for e in self.cv_t1_entries]
        self.t1.points = values[0]
        self.t1.spacing = values[1]
        self.t1.tau_max = values[2]
        self.t1.pi_pulse_length = values[3]
        self.t1.mw_frequency = values[4]
        self.t1.mw_amplitude = values[5]
        self.t1.laser_spot_position = float(self.laser_spot.get())
        begin_potential = float(values[6])
        end_potential = float(values[7])
        step_potential = float(values[8])/1000
        scanrate = float(values[9])/1000
        method = self.potentiostat.cv_measurement(begin_potential, end_potential, step_potential, scanrate)
        ec_points = (abs(begin_potential-end_potential)/step_potential)*2
        ec_time = step_potential/scanrate
        avg_p_p = int(values[10])
        self.t1.averages = calculate_t1_averages(ec_points, avg_p_p, ec_time, int(values[0]), float(values[1]))
        user_params = self.t1.export_params()
        self.qs_file_name, self.ec_file_name = create_file_name(qs_type="T1",ec_type="CV",annotation=values[10])
        self.current_measurement_method = "CV-T1"
        
        Send_t1_measurement(self.qs_file_name, user_params)
        self.potentiostat.start_measurement_until_stop(self.ec_file_name, method,avg_p_p)
        self.qs_data_reader = QS_data_reader(self.data_folder, self.qs_file_name)
        self.qs_data_reader.start_data_building()
        self.show_frame("live")
        self.update_ecqs_live_figure()
    def on_swv_t1_send(self):
        self.t1 = User_T1_input()
        self.cleanup_measurement_state()
        self.potentiostat = Potentiostat(self.data_folder)
        self.ecqs_cache = ECQS_cache()
        """
        "Points:",
        "Max. \u03C4 [ms]:",
        "\u03C0-pulse Length [ns]:",
        "MW Frequency [Ghz]:",
        "MW Amplitude [dB]:",
        "Start Potential [V]:",
        "End Potential [V]:",
        "Potential Step [mV]:",
        "Frequency [Hz]:",
        "Amplitude [mV]",
        "T1-averages / SWV-point:",
        "Annotation:"
        """
        values = [e.get() for e in self.swv_t1_entries]
        self.t1.points = values[0]
        self.t1.spacing = values[1]
        self.t1.tau_max = values[2]
        self.t1.pi_pulse_length = values[3]
        self.t1.mw_frequency = values[4]
        self.t1.mw_amplitude = values[5]
        self.t1.laser_spot_position = float(self.laser_spot.get())
        begin_potential = float(values[6])
        end_potential = float(values[7])
        step_potential = float(values[8])/1000
        frequency = float(values[9])
        amplitude = float(values[10])/1000
        method = self.potentiostat.swv_measurement(begin_potential, end_potential, step_potential, frequency, amplitude)
        avg_p_p = int(values[11])
        ec_points = (abs(begin_potential-end_potential)/step_potential)
        ec_time = 1/frequency
        self.t1.averages = calculate_t1_averages(ec_points, avg_p_p, ec_time, int(values[0]), float(values[1]))
        user_params = self.t1.export_params()
        self.qs_file_name, self.ec_file_name = create_file_name(qs_type="T1",ec_type="SWV",annotation=values[11])
        self.current_measurement_method = "SWV-T1"
        
        Send_t1_measurement(self.qs_file_name, user_params)
        self.potentiostat.start_measurement_until_stop(self.ec_file_name, method,avg_p_p)
        self.qs_data_reader = QS_data_reader(self.data_folder, self.qs_file_name)
        self.qs_data_reader.start_data_building()
        self.show_frame("live")
        self.update_ecqs_live_figure()
        
    def on_lsv_t1_send(self):
        self.t1 = User_T1_input()
        self.cleanup_measurement_state()
        self.potentiostat = Potentiostat(self.data_folder)
        self.ecqs_cache = ECQS_cache()
        """
        [ "Points:",
         "Max. \u03C4 [ms]:",
         "\u03C0-pulse Length [ns]:",
         "MW Frequency [Ghz]:",
         "MW Amplitude [dB]:",
         "Start Potential [V]:","End Potential [V]:","Potential Step [mV]:","Scanrate [mV/s]:","T1-averages / LSV-point:","Annotation:"]
        """
        values = [e.get() for e in self.lsv_t1_entries]
        self.t1.points = values[0]
        self.t1.spacing = values[1]
        self.t1.tau_max = values[2]
        self.t1.pi_pulse_length = values[3]
        self.t1.mw_frequency = values[4]
        self.t1.mw_amplitude = values[5]
        self.t1.laser_spot_position = float(self.laser_spot.get())
        begin_potential = float(values[6])
        end_potential = float(values[7])
        step_potential = float(values[8])/1000
        scanrate = float(values[9])/1000
        method = self.potentiostat.lsv_measurement(begin_potential, end_potential, step_potential, scanrate)
        avg_p_p = int(values[10])
        ec_points = (abs(begin_potential-end_potential)/step_potential)
        ec_time = step_potential/scanrate
        self.t1.averages = calculate_t1_averages(ec_points, avg_p_p, ec_time, int(values[0]), float(values[1]))
        user_params = self.t1.export_params()
        self.qs_file_name, self.ec_file_name = create_file_name(qs_type="T1",ec_type="LSV",annotation=values[10])
        self.current_measurement_method = "LSV-T1"
        Send_t1_measurement(self.qs_file_name, user_params)
        self.potentiostat.start_measurement_until_stop(self.ec_file_name, method,avg_p_p)
        self.qs_data_reader = QS_data_reader(self.data_folder, self.qs_file_name)
        self.qs_data_reader.start_data_building()
        self.show_frame("live")
        self.update_ecqs_live_figure()
 
    def update_qs_live_figure(self):
        current_fig = self.live_canvas.figure if self.live_canvas is not None else None
        self.fig, _, measurement_done, prog, info = qs_data_displayer(self.qs_data_reader, {"fitting": False}, fig=current_fig)
        self.measurement_info_update(info)
        self.live_progress['value'] = prog * 100
        self.live_progress.update_idletasks()
        
        if self.live_canvas is None:
            self.live_canvas = FigureCanvasTkAgg(self.fig, master=self.live_plot_container)
            self.live_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.live_canvas.draw()
        
        if not measurement_done:
            self.live_status_label.config(text=f"Running: {self.qs_file_name}")
            self.live_update_job = self.after(200, self.update_qs_live_figure)
        else:
            was_cancelled = self.cancel_requested
            self.live_update_job = None
            if was_cancelled:
                self.live_status_label.config(text=f"Stopped: {self.qs_file_name}")
            else:
                self.live_status_label.config(text=f"Finished: {self.qs_file_name}")
            self.eval_qs_figure()
            self.show_frame(["evaluator", "measure"])
            self.update_file_loader()
            self.finalize_cancelled_measurement()
    def _replace_final_canvas(self):
        if self.final_canvas is not None:
            old_fig = self.final_canvas.figure
            self.final_canvas.get_tk_widget().destroy()
            self.final_canvas = None
            if old_fig is not None and old_fig is not self.fig:
                plt.close(old_fig)

        self.final_canvas = FigureCanvasTkAgg(self.fig, master=self.final_plot_container)
        self.final_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.final_canvas.draw()
    def eval_qs_figure(self, fitting_params:dict = None):
       
        if fitting_params == None:
            self.fig,_,_,_,info = qs_data_displayer(self.qs_data_reader, {"fitting":True})
        else:
            self.fig,_,_,_,info = qs_data_displayer(self.qs_data_reader,fitting_params)
        self._replace_final_canvas()

        self.fitting_parameter_update()    
        self.measurement_info_update(info)
        self.update_file_loader()
    def update_ec_live_figure(self):    
        current_fig = self.live_canvas.figure if self.live_canvas is not None else None
        self.fig, _, measurement_done, prog, info = ec_data_displayer(self.potentiostat,{"fitting": False}, fig=current_fig)
        self.measurement_info_update(info)
        self.live_progress['value'] = prog*100
        self.live_progress.update_idletasks()  # force GUI to refresh
        
        if self.live_canvas is None:
            self.live_canvas = FigureCanvasTkAgg(self.fig, master=self.live_plot_container)
            self.live_canvas.get_tk_widget().pack(fill="both", expand=True)
        # Update status
        if not measurement_done:
            self.live_status_label.config(text=f"Running measurement: {self.ec_file_name}")
        else:
            if self.cancel_requested:
                self.live_status_label.config(text=f"Stopped measurement: {self.ec_file_name}")
            else:
                self.live_status_label.config(text=f"Finished measurement: {self.ec_file_name}")
            #self.live_progress.pack_forget()
        self.live_canvas.draw()
    
        if not measurement_done:
            self.live_update_job = self.after(200, self.update_ec_live_figure)
        else:
            self.live_update_job = None
            self.eval_ec_figure()
            self.show_frame(["evaluator","measure"])
            self.update_file_loader()
            self.finalize_cancelled_measurement()
    def eval_ec_figure(self,fitting_params:dict = None):
        if fitting_params == None:
            self.fig,_,_,_,info = ec_data_displayer(self.potentiostat, {"fitting":True})
        else:
            self.fig,_,_,_,info = ec_data_displayer(self.potentiostat,fitting_params)
        self._replace_final_canvas()

        self.fitting_parameter_update()    
        self.measurement_info_update(info)
        self.update_file_loader()
        
    def update_ecqs_live_figure(self):    
        current_fig = self.live_canvas.figure if self.live_canvas is not None else None
        self.fig, _, measurement_done, prog, info, kill_measurement = ecqs_data_displayer(self.qs_data_reader,self.potentiostat,{"fitting": False}, self.ecqs_cache, fig=current_fig)
        if not np.isfinite(prog):
            prog = 0
        self.measurement_info_update(info)
        self.live_progress['value'] = prog*100
        self.live_progress.update_idletasks()  # force GUI to refresh
        if self.live_canvas is None:
            self.live_canvas = FigureCanvasTkAgg(self.fig, master=self.live_plot_container)
            self.live_canvas.get_tk_widget().pack(fill="both", expand=True)
        # Update status
        if kill_measurement ==True and not self.potentiostat.stop_measurement.is_set():
            self.cancel_measurement()
        if not measurement_done:
            self.live_status_label.config(text=f"Running measurement: {self.ec_file_name}")
        else:
            if self.cancel_requested:
                self.live_status_label.config(text=f"Stopped measurement: {self.ec_file_name}")
            else:
                self.live_status_label.config(text=f"Finished measurement: {self.ec_file_name}")
            #self.live_progress.pack_forget()
        self.live_canvas.draw()
    
        if not measurement_done:
            self.live_update_job = self.after(200, self.update_ecqs_live_figure)
        else:
            self.live_update_job = None
            self.show_frame(["evaluator","measure"])
            self.update_file_loader()
            self.finalize_cancelled_measurement()
            self.eval_ecqs_figure()
    def eval_ecqs_figure(self,fitting_params:dict = None):
        self.ecqs_cache = ECQS_cache()
        if fitting_params == None:
            self.fig,_,_,_,info,_ = ecqs_data_displayer(self.qs_data_reader,self.potentiostat, {"fitting":True}, self.ecqs_cache)
        else:
            self.fig,_,_,_,info,_ = ecqs_data_displayer(self.qs_data_reader,self.potentiostat,fitting_params, self.ecqs_cache)
        self._replace_final_canvas()

        self.fitting_parameter_update()    
        self.measurement_info_update(info)
        self.update_file_loader()
        
    def save_figure(self):
       file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG image", "*.png"),
            ("PDF file", "*.pdf"),
            ("SVG file", "*.svg"),
            ("All files", "*.*")
        ],
        title="Save figure as...")
       if file_path:  # user didn't cancel
            self.fig.savefig(file_path, dpi=300, bbox_inches="tight")
            print("Saved to:", file_path)     
    def esr_fitting(self):
        
        values = [self.get_fitting_value(item) for item in self.esr_fitting_entries]
        fit_dict = {"fitting":values[0],
                    "esr_prominence": values[1],
                    "esr_distance":values[2]
            }
        print(values)
        self.eval_qs_figure(fit_dict)
    def rabi_fitting(self):
        # ["Fitting:","Use Sinus_fit:","Show Sinus fit:","Use Bounds:","Max. Tries:","Use Peakfinding:","Prominence:","Distance:"]
        values = [self.get_fitting_value(item) for item in self.rabi_fitting_entries]
        fit_dict = {"fitting":values[0],
                    "rabi_sin_fit":values[1],
                    "rabi_show_fit":values[2],
                    "rabi_use_bounds":values[3],
                    "rabi_max_tries":values[4],
                    "rabi_find_peak":values[5],
                    "rabi_prominence":values[6],
                    "rabi_distance":values[7]
            }
        print(values)
        self.eval_qs_figure(fit_dict)
    def t1_fitting(self):
        #"Fitting:","Use stretched exponential fit:","Stretched exp. fit - Max. tries:","Stretched exp. fit - use Bounds:","Show stretched exp. fit:",
        #"Use Biexpontential fit:","biexp. fit - Max. tries:","biexp.fit - use Bounds:","Show biexp. fit:"
        values = [self.get_fitting_value(item) for item in self.t1_fitting_entries]
        fit_dict = {"fitting" : values[0],
                    "t1_stretched_exp_fit":values[1],
                    "t1_max_stretch_tries":values[2],
                    "t1_use_stretch_bounds":values[3],
                    "t1_show_stretch_fit":values[4],
                    "t1_bi_exp_fit":values[5],
                    "t1_max_bi_tries":values[6],
                    "t1_use_bi_bounds":values[7],
                    "t1_show_bi_fit":values[8]
            }
        print(values)
        self.eval_qs_figure(fit_dict)
    def cv_fitting(self):
        values = [self.get_fitting_value(item) for item in self.cv_fitting_entries]
        fit_dict = {"fitting":values[0],
                    "prominence": values[1],
                    "distance":values[2]
            }
        print(values)
        self.eval_ec_figure(fit_dict)
    def swv_fitting(self):
        values = [self.get_fitting_value(item) for item in self.swv_fitting_entries]
        fit_dict = {"fitting":values[0],
                    "prominence": values[1],
                    "distance":values[2]
            }
        print(values)
        self.eval_ec_figure(fit_dict)
    def lsv_fitting(self):
        values = [self.get_fitting_value(item) for item in self.lsv_fitting_entries]
        fit_dict = {"fitting":values[0],
                    "prominence": values[1],
                    "distance":values[2]
            }
        print(values)
        self.eval_ec_figure(fit_dict)
    
    def cv_t1_fitting(self):
        #["Fitting:","Prominence:","Distance:","Cutoff Rate","Use Bounds:","Max. Tries:"]
        values = [self.get_fitting_value(item) for item in self.cv_t1_fitting_entries]
        fit_dict = {"ec_fitting":values[0],
                    "cutoff_rate":values[3],
                    "max_tries":values[5],
                    "use_bounds":values[4],
                    "prominence":values[1],
                    "distance":values[2]}
        print(values)
        self.eval_ecqs_figure(fit_dict)
    def swv_t1_fitting(self):
        #["Fitting:","Prominence:","Distance:","Cutoff Rate","Use Bounds:","Max. Tries:"]
        values = [self.get_fitting_value(item) for item in self.swv_t1_fitting_entries]
        fit_dict = {"ec_fitting":values[0],
                    "cutoff_rate":values[3],
                    "max_tries":values[5],
                    "use_bounds":values[4],
                    "prominence":values[1],
                    "distance":values[2]}
        print(values)
        self.eval_ecqs_figure(fit_dict)
    def lsv_t1_fitting(self):
        #["Fitting:","Prominence:","Distance:","Cutoff Rate","Use Bounds:","Max. Tries:"]
        values = [self.get_fitting_value(item) for item in self.lsv_t1_fitting_entries]
        fit_dict = {"ec_fitting":values[0],
                    "cutoff_rate":values[3],
                    "max_tries":values[5],
                    "use_bounds":values[4],
                    "prominence":values[1],
                    "distance":values[2]}
        print(values)
        self.eval_ecqs_figure(fit_dict)
    def reset_live_canvas(self):
        self.cancel_live_update_job()
        if self.live_canvas is not None:
            old_fig = self.live_canvas.figure
            self.live_canvas.get_tk_widget().destroy()
            self.live_canvas = None
            if old_fig:
                plt.close(old_fig)
App().mainloop()
