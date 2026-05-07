# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:58:11 2026

@author: Benno Dauer
"""

import yaml
import os
import time
import subprocess
#this will send measurement yaml files from the protocol folder to the waiting room 
#will also have fixes params set, independent od the yaml file and allow for input to be set
#Never use send_measurement, only the subclasses
class Send_measurement():
    def __init__(self, file_name, user_params: dict = {}) -> str:
        self.file_name = file_name
        self.waiting_room_path = "C:/Users/ge54vec/.qupyt/waiting_room"
        self.param_file = f"G:/Bucherlab/Surface_Electrochemistry_EPR/Protocols/{self.measurement_type}.yaml"
        self.old_params = self.write_params_from_yaml(self.param_file)
        self.new_params = user_params
        self.fixed_params = { "file_name" : self.file_name,
            "data":{"temporary_saving": True},
            "experiment_type" :f"{self.measurement_type}"
                              }
        self.params = self.overwrite_params_dic(self.old_params, self.new_params)
        self.params = self.overwrite_params_dic(self.params, self.fixed_params)
        self.params =  self.overwrite_params_dic(self.params, self.specific_fixed_params)
        
        self.send_to_queue()
        return file_name
    
    def write_params_from_yaml(self, yaml_file : str) -> dict:
        with open(yaml_file, "r") as f:
            params = yaml.safe_load(f)
        return params
    
    def overwrite_params_dic(self, old_dic: dict, new_dic: dict) -> dict:
        return_dic = old_dic.copy()
        for key,val in new_dic.items():
            if isinstance(val, dict) and isinstance(return_dic.get(key), dict):
                return_dic[key] = self.overwrite_params_dic(return_dic[key], val)
            else:
                return_dic[key] = val
        return return_dic
    
    def send_to_queue(self) -> None:
        with open(f"{self.waiting_room_path}/{self.file_name}.yaml", "w") as file:
            yaml.dump(self.params, file)

class Send_esr_measurement(Send_measurement):
    def __init__(self, file_name, user_params = {}):
        self.measurement_type = "ESR_ECQS"
        self.specific_fixed_params = {}
        super().__init__( file_name, user_params)    
        
class Send_rabi_measurement(Send_measurement):
    def __init__(self, file_name, user_params = {}):
        self.measurement_type = "Rabi_ECQS"
        self.specific_fixed_params = {}
        super().__init__( file_name, user_params)    
        
class Send_t1_measurement(Send_measurement):
    def __init__(self, file_name, user_params = {}):
        self.measurement_type = "T1_ECQS"
        self.specific_fixed_params = {}
        super().__init__( file_name, user_params)    
        
class Send_laser_off(Send_measurement):
    def __init__(self):
        self.measurement_type = "Laser_off"
        self.specific_fixed_params = {}
        super().__init__("laser_off")
        
        
#this class will close and open qupyt. there is a waiting period of 7 sec after opening qupyt to boot. after init it will close if qupyt was still open and delete the waiting room
class Qupyt_controller():
    def __init__(self, measurement_path: str) -> None:
        self.waiting_room_path = "C:/Users/ge54vec/.qupyt/waiting_room/"
        self.measurement_path = measurement_path
        self.exit()
        
    #Start qupyt
    def start(self):
        os.system(f'start /D "C:/Users/ge54vec/python_env/env_qupyt/Scripts" powershell -NoExit -Command "& .\Activate.ps1; Set-Location \\"{self.measurement_path}\\"''; qupyt --verbose"')
        #time.sleep(7)
        
        # Exit Qupyt and close the powershell; delete every file, which is in the waiting room
    def exit(self):
         #this does not work currently   
        #os.system("taskkill /f /fi \WindowTitle eq C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\'")
        subprocess.run(
            ["taskkill", "/F", "/IM", "powershell.exe", "/T"],
            shell=True
                )
        if os.path.isdir(self.waiting_room_path):
            for file_name in os.listdir(self.waiting_room_path):
                folder_to_delete = os.path.join(self.waiting_room_path, file_name)
                os.remove(folder_to_delete) 
        
#This class and its subclasses read and create the yaml files for the qs measurements    
class User_parameter_input():
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_path = "G:/Bucherlab/Surface_Electrochemistry_EPR/Protocols/"
        self.params = self.write_params_from_yaml(os.path.join(self.file_path, self.file_name))
    def write_params_from_yaml(self, yaml_file : str) -> dict:
        with open(yaml_file, "r") as f:
            params = yaml.safe_load(f)
        return params
    

class User_ESR_input(User_parameter_input):
    def __init__(self):
        self.file_name = "ESR_ECQS.yaml"
        super().__init__(self.file_name)
        self.points = int(self.params["dynamic_steps"])
        self.averages =  int(self.params['sensor']['config']['number_measurements'])
        self.min_mw_freq = float(self.params['dynamic_devices']['mw_source']['config']['frequency'][0][1][0])/1000000000
        self.max_mw_freq = float(self.params['dynamic_devices']['mw_source']['config']['frequency'][0][1][1])/1000000000
        self.mw_amplitude = int(self.params['dynamic_devices']['mw_source']['config']['amplitude'][0][1][0])
        self.laser_spot_position = int(self.params['dynamic_devices']['mw_source']['config']['frequency'][1][1][0])/1000000
    def export_params(self) -> dict:
        export = self.params
        export['dynamic_devices']['mw_source']['config']['frequency'][1][1][0] = float(self.laser_spot_position)*1000000
        export['dynamic_devices']['mw_source']['config']['frequency'][1][1][1] = float(self.laser_spot_position)*1000000
        export['dynamic_devices']['mw_source']['config']['amplitude'][0][1][0] = float(self.mw_amplitude)
        export['dynamic_devices']['mw_source']['config']['amplitude'][0][1][1] = float(self.mw_amplitude)
        export['dynamic_devices']['mw_source']['config']['frequency'][0][1][0] = float(self.min_mw_freq)*1000000000
        export['dynamic_devices']['mw_source']['config']['frequency'][0][1][1] = float(self.max_mw_freq)*1000000000
        export['sensor']['config']['number_measurements'] = int(self.averages)
        export['pulse_sequence']['number_measurements'] = int(self.averages)
        export["dynamic_steps"] = int(self.points)
        export["data"]["dynamic_steps"] = int(self.points)
        return export
    
class User_Rabi_input(User_parameter_input):
    def __init__(self):
        self.file_name = "Rabi_ECQS.yaml"
        super().__init__(self.file_name)
        self.max_pulse_duration = int(self.params["sensor"]["config"]["number_measurements"])
        self.averages = int(self.params["averages"])
        self.mw_frequency = float(self.params['static_devices']['mw_source']['config']['frequency'][0][1])/1000000000
        self.mw_amplitude = float(self.params['static_devices']['mw_source']['config']['amplitude'][0][1])
        self.laser_spot_position = int(self.params['static_devices']['mw_source']['config']['frequency'][1][1])/1000000
    def export_params(self) -> dict:
        export = self.params
        export["sensor"]["config"]["number_measurements"] = int(self.max_pulse_duration)
        export['pulse_sequence']['rabi_steps'] = int(self.max_pulse_duration)
        export["averages"] = int(self.averages)
        export['static_devices']['mw_source']['config']['frequency'][0][1] = float(self.mw_frequency)*1000000000
        export['static_devices']['mw_source']['config']['amplitude'][0][1] = float(self.mw_amplitude)
        export['static_devices']['mw_source']['config']['frequency'][1][1] = int(self.laser_spot_position)*1000000
        return export
    
class User_T1_input(User_parameter_input):
    def __init__(self):
        self.file_name ="T1_ECQS.yaml"
        super().__init__(self.file_name)
        self.points = int(self.params["sensor"]["config"]["number_measurements"]/2)
        self.averages = int(self.params["averages"])
        self.tau_max = float(self.params["pulse_sequence"]["t1_max"])/1000
        self.pi_pulse_length = float(self.params['pulse_sequence']['mw_duration']) *1000
        self.mw_frequency = float(self.params['static_devices']['mw_source']['config']['frequency'][0][1])/1000000000
        self.mw_amplitude = float(self.params['static_devices']['mw_source']['config']['amplitude'][0][1])
        self.laser_spot_position = int(self.params['static_devices']['mw_source']['config']['frequency'][1][1])/1000000
    def export_params(self):
        export = self.params
        export["sensor"]["config"]["number_measurements"] = int(self.points)
        export["pulse_sequence"]["t1_steps"] = int(self.points)*2
        export['sensor']['config']['number_measurements'] = int(self.points)*2
        export["averages"] = int(self.averages)
        export["pulse_sequence"]["t1_max"] = float(self.tau_max)*1000
        export['static_devices']['mw_source']['config']['frequency'][0][1] = float(self.mw_frequency)*1000000000
        export['static_devices']['mw_source']['config']['amplitude'][0][1] = float(self.mw_amplitude)
        export['static_devices']['mw_source']['config']['frequency'][1][1] = int(self.laser_spot_position)*1000000
        export['pulse_sequence']['mw_duration'] = float(self.pi_pulse_length)/1000
        return export
        