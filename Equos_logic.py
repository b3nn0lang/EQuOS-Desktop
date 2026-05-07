# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:11:49 2026

@author: Benno Dauer
"""


from datetime import datetime
import os
import numpy as np
import math
#This method creates the file names for the measurements in a way that parse_file_folder sees which ones belong together
def create_file_name(qs_type: str = None, ec_type: str = None, annotation: str = "") -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    ec_file_name = ""
    qs_file_name = ""

    # guard clause - check this FIRST
    if ec_type is None and qs_type is None:
        raise Exception("No experimental type was given")

    annotation_str = f"_{annotation}" if annotation != "" else ""

    if ec_type is None:
        qs_file_name = f"QS_{qs_type}_{timestamp}{annotation_str}"

    elif qs_type is None:
        ec_file_name = f"EC_{ec_type}_{timestamp}{annotation_str}"

    else:  # both given
        qs_file_name = f"ECQS_{qs_type}_{timestamp}{annotation_str}"
        ec_file_name = f"ECQS_{ec_type}_{timestamp}{annotation_str}"

    return qs_file_name, ec_file_name
#This method parses the current folder to load data and returns a dicteniorray with the splitted filename parts (only works is solely equos files are present)
def parse_file_folder(folder_name: str) -> list:
       measurements = list()
       file_name_list = set()
       for file in os.listdir(folder_name):
           file = file.replace("_temp.dat","").replace(".yaml","").replace(".pssession","").replace(".csv","")
           yaml_file = file +".yaml"
           dat_file = file+"_temp.dat"
           csv_file = file+".csv"
           pssession_file=file+".pssession"
           full_yaml_path = os.path.join(folder_name,yaml_file)
           full_dat_path = os.path.join(folder_name, dat_file)
           full_csv_path = os.path.join(folder_name,csv_file)
           full_psssession_path=os.path.join(folder_name, pssession_file)
           if (os.path.isfile(full_dat_path) and os.path.isfile(full_yaml_path)) or (os.path.isfile(full_csv_path) and os.path.isfile(full_psssession_path)):
               file_name_list.add(file)
       file_name_list = sorted(file_name_list)
       for file in file_name_list:
           params = file.split("_")
           exp_type = params[0] if len(params) > 0 else ""
           method_type = params[1] if len(params) > 1 else ""
           date = params[2] if len(params) > 2 else ""
           annotation = "_".join(params[3:]) if len(params) > 3 else ""
           dt = datetime.strptime(date, "%Y%m%d%H%M%S")
           measurements.append({
                "file_name": file,
                "type": exp_type,
                "method": method_type,
                "date_raw": date,
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M:%S"),
                "annotation": annotation
            })
       return measurements
           
#calculeates the t1 averages for the ecqs measuerements form measurements per point           
def calculate_t1_averages(ec_points:int,avg_p_p:int, ec_time:float, qs_points:int, qs_max_tau:float) ->float:
   safety_factor = 5 #this is tunable
   
   ec_run_time = ec_time*ec_points #total time for one ec measurement
   qs_avg_time = (np.sum(np.logspace(np.log10(0.2/1000), np.log10(qs_max_tau), num=qs_points)+25/1000 + 5))/1000 # total time for one t1 average wutg saving 
   ec_runs = math.ceil((qs_avg_time/ec_time)*avg_p_p) 
   total_time = ec_runs*ec_run_time
   qs_averages = total_time/qs_avg_time
   qs_final = qs_averages*safety_factor
   print("ec_run_time,qs_avg_time,ec_runs,total_time,qs_averages,qs_final: ",ec_run_time,qs_avg_time,ec_runs,total_time,qs_averages,qs_final)
   return qs_final
    
           
class ECQS_cache():
    def __init__(self):
        self.last_ec_data = None
        self.fig  = None
        self.axes = None
        self.measurement_done = None
        self.measurement_progress = None
        self.info_dic = None
        self.kill_qupyt = None
    def compare_last_values(self, new_ec_data) -> bool:
        if self.last_ec_data is None:
            return True
        try:
            return not new_ec_data.equals(self.last_ec_data)
        except Exception:
            return True
    def set_values(self, fig, axes, measurement_done, measurement_progress, info_dic, kill_qupyt):
        self.fig  = fig
        self.axes = axes
        self.measurement_done = measurement_done
        self.measurement_progress = measurement_progress
        self.info_dic = info_dic
        self.kill_qupyt = kill_qupyt
    def get_values(self):
        return self.fig, self.axes, self.measurement_done, self.measurement_progress, self.info_dic, self.kill_qupyt
        
        
    

    

        
        


