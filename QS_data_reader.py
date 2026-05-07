#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:22:07 2026

@author: bennodauer
"""

import numpy as np
import pandas as pd
import yaml
import time
import os
import shutil
import threading

#This class gets the data file path and name and will write the raw data to a pandas frame
class QS_data_reader():
      
    def __init__(self, file_path: str,file_name: str) -> None:
        self._dataframe_lock = threading.RLock()
        self._progress_lock = threading.RLock()
        self.dataframe_loop_stop = threading.Event()
        self.raw_dataframe_progress = 0
        self.scanning_interval = 0.5
        self.qupyt_is_dead_time = 3
        self.qupyt_param_creation_time = 10
        self.temp_path = "C:/Users/ge54vec/.qupyt/temp"
        self.file_path = file_path
        self.file_name = file_name
        
        try:
            for i in range(100):
                time.sleep(self.qupyt_param_creation_time/100)
                if os.path.isfile(f"{self.file_path}/{self.file_name}.yaml"):
                    self.params = self.write_params_from_yaml(f"{self.file_path}/{self.file_name}.yaml")
                    print("params found")
                    break
                else:
                    print(f"Try [{i}/100] to find params")
        except Exception as e:
            raise FileNotFoundError(f"{self.file_path}/{self.file_name}.yaml has not been found: {e}")
        try:    
            self.experiment_type = str(self.params["experiment_type"])
            self.averaging_mode = self.params["data"]["averaging_mode"]
            self.reference_channels = int(self.params["data"]["reference_channels"])
            self.number_dynamic_steps = int(self.params["data"]["dynamic_steps"])
            self.number_measurements = int(self.params["sensor"]["config"]["number_measurements"])
            self.averages = int(self.params["averages"])
            if self.experiment_type == "T1_ECQS" or self.experiment_type == "T1":
                #self.x_values = list(self.params["tau_spacings"])
                self.x_values = list([float(tau) / 1000 for tau in self.params["tau_spacings"]])
                self.x_type = "tau spacing"
                self.x_unit = "ms"
            elif self.experiment_type == "ESR_ECQS" or self.experiment_type == "ESR":
                min_freq = float(self.params['dynamic_devices']['mw_source']['config']['frequency'][0][1][0])/1000000000 # load min and max frequency to calculate the x axis
                max_freq = float(self.params['dynamic_devices']['mw_source']['config']['frequency'][0][1][1])/1000000000
                dyn_steps = int(self.params['dynamic_steps']) # load dynamic steps to reshape array
                self.x_values = list(np.linspace(min_freq,max_freq,dyn_steps))
                self.x_type = "mw pulse frequency"
                self.x_unit = "Ghz"
            elif self.experiment_type == "Rabi_ECQS" or self.experiment_type == "Rabi":
                self.x_values = list(self.params["pi_pulse_durations"])
                self.x_type = "pi pulse duration"
                self.x_unit = "ns"
        except Exception as e:
            print(f"{self.experiment_type}")
            raise KeyError(f"One of the parameters in {self.file_name}.yaml has not been defined. Have fun looking: {e}")
            

        #self.raw_dataframe = pd.DataFrame()
        self._dataframe_consolidated = pd.DataFrame()
        self.data_array_dim = self.build_data_file_dimension()
        self._processed_rows = 0

#Evalutes the params.yaml file        
    def write_params_from_yaml(self, yaml_file : str) -> dict:
        with open(yaml_file, "r") as f:
            params = yaml.safe_load(f)
        return params

#replicate the data file, that qupyt will output if temp is activated, only works if tempp is activated
    def build_data_file_dimension(self: str) -> np.shape:
        
        mmap_width = 2+self.number_measurements
        mmap_length = self.averages * self.number_dynamic_steps
        data_array_dim = (mmap_length, mmap_width)
        print(f"data_array_dim = {data_array_dim}")
        return data_array_dim
     #checks for the file in the qupyt temp folder and the measurement folder and returns the status   
    """
    def scan_data_files(self):
        try:
            if os.path.isfile(f"{self.file_path}/{self.file_name}_temp.dat"):
                print("temp.dat exists in file folder - measurement finished")
                current_data = self.read_data_file(f"{self.file_path}/{self.file_name}_temp.dat")
                measurement_finished = True             
                
            elif os.path.isfile(f"{self.temp_path}/{self.file_name}_temp.dat"):
                #print("temp.dat exists in temp folder")
                current_data = self.read_data_file(f"{self.temp_path}/{self.file_name}_temp.dat")
                #print(f"current_data_dimensions are {current_data.shape}, raw_data dimensions are {self.raw_data.shape}")
                if (current_data == self.raw_data).all():
                    for i in range(10):
                        print(f"No new data points - waiting for qupyt [{i}/10]")
                        time.sleep(self.qupyt_is_dead_time/10)
                        current_data = self.read_data_file(f"{self.temp_path}/{self.file_name}_temp.dat")
                        if (current_data == self.raw_data).all():
                            print("Assuming measuerement interruption. make sure qupyt is not running")
                            measurement_finished = True
                            if i == 9:
                                print("moving measurement from temp to file folder")
                                time.sleep(0.1)
                                shutil.copy(os.path.join(self.temp_path,f"{self.file_name}_temp.dat"), os.path.join(self.file_path,f"{self.file_name}_temp.dat"))
                                time.sleep(0.1)
                                os.remove(os.path.join(self.temp_path,f"{self.file_name}_temp.dat"))
                        else:
                            current_data = self.read_data_file(f"{self.temp_path}/{self.file_name}_temp.dat")
                            measurement_finished = False
                            break
                else: 
                    measurement_finished = False
            else:
                raise FileNotFoundError(f"Error: {self.file_name}_temp.dat not found")
            return current_data, measurement_finished
        except Exception as e:
            raise Exception(e)
       """
    def scan_data_files(self):
        try:
            if os.path.isfile(f"{self.file_path}/{self.file_name}_temp.dat"):
                print("temp.dat exists in file folder - measurement finished")
                dat_file = f"{self.file_path}/{self.file_name}_temp.dat"
                return dat_file, self.count_filled_rows(dat_file), True
    
            elif os.path.isfile(f"{self.temp_path}/{self.file_name}_temp.dat"):
                dat_file = f"{self.temp_path}/{self.file_name}_temp.dat"
                current_rows = self.count_filled_rows(dat_file)
    
                if current_rows != self._processed_rows:
                    return dat_file, current_rows, False
    
                for i in range(10):
                    print(f"No new data points - waiting for qupyt [{i}/10]")
                    time.sleep(self.qupyt_is_dead_time / 10)
                    new_rows = self.count_filled_rows(dat_file)
                    if new_rows != current_rows:
                        return dat_file, new_rows, False
                    current_rows = new_rows
    
                print("Assuming measurement interruption. Moving file to measurement folder.")
                time.sleep(0.1)
                shutil.copy(
                    os.path.join(self.temp_path, f"{self.file_name}_temp.dat"),
                    os.path.join(self.file_path, f"{self.file_name}_temp.dat")
                )
                time.sleep(0.1)
                os.remove(os.path.join(self.temp_path, f"{self.file_name}_temp.dat"))
                return os.path.join(self.file_path, f"{self.file_name}_temp.dat"), current_rows, True
    
            else:
                raise FileNotFoundError(f"Error: {self.file_name}_temp.dat not found")
    
        except Exception as e:
            raise Exception(e)

    
    
    
    #reads the memmap .dat file   
    def count_filled_rows(self, dat_file: str) -> int:
        mm = np.memmap(dat_file, dtype=float, mode="r", shape=self.data_array_dim)
        try:
            return int(np.count_nonzero(np.any(mm[:, :2] != 0, axis=1)))
        finally:
            del mm

    #This iteratively builds the dataframe with raw data, depending on the measuemrent mode: sum or spread and the x_axis type
    def build_dataframe(self, dat_file: str, filled_rows: int) -> None:
        current_len = self._processed_rows
        new_range = filled_rows - current_len
        if new_range < 0:
            raise EOFError("Dataframe longer than raw Data")
        elif new_range == 0:
            pass
        else:
            mm = np.memmap(dat_file, dtype=float, mode="r", shape=self.data_array_dim)
            try:
                new_rows = mm[current_len:filled_rows]
                t_begin = np.array(new_rows[:, 0], copy=True)
                t_end = np.array(new_rows[:, 1], copy=True)
                signal_rows = np.array(new_rows[:, 2:], copy=True)
            finally:
                del mm

            channel_dic = {
                ch: [row[ch::self.reference_channels].copy() for row in signal_rows]
                           for ch in range(self.reference_channels)
                           }
            contrast = list(np.divide(channel_dic[1], channel_dic[0]))
            if self.averaging_mode == "spread" and self.x_values != []:
                x_values = [self.x_values for i in range(new_range)]
            elif self.averaging_mode == "sum" and self.x_values != []:
                x_values = self.x_values[current_len:(current_len+new_range)]
                #print(new_range)
                #print(x_values)
            else:
                raise Exception("Averaging mode is neither spread nor sum or x_values not known")
                
            df_to_append = pd.DataFrame(
                {"t_begin": t_begin ,
                 "t_end": t_end,
                 #"raw_data": raw_data,
                 **{f"channel_{key}": val for key,val in channel_dic.items()},
                 "contrast" : contrast,
                 f"{self.x_type}": x_values
                  }
                )
            
            with self._dataframe_lock:
                if self._dataframe_consolidated.empty:
                    self._dataframe_consolidated = df_to_append.reset_index(drop=True)
                else:
                    start_idx = len(self._dataframe_consolidated)
                    idx = range(start_idx, start_idx + len(df_to_append))
                    self._dataframe_consolidated.loc[idx, df_to_append.columns] = df_to_append.to_numpy(dtype=object)
            self._processed_rows = filled_rows
    #use this to access the dataframe to avoid problems with multithreading
    def get_dataframe(self) -> pd.DataFrame:
        with self._dataframe_lock:
            return self._dataframe_consolidated.copy()
    # this is the main loop to be activated that will then build the DF. It works with threading, so in parallel to the main script
    def start_data_building(self):
        data_loop = threading.Thread(target= self.data_building_loop, daemon=True)
        data_loop.start()
    #this is the actual loop
    def data_building_loop(self ) -> None:
        self.dataframe_loop_stop.clear()
        while not self.dataframe_loop_stop.is_set(): 
            dat_file, filled_rows, status = self.scan_data_files()
            self.build_dataframe(dat_file, filled_rows)
            #print(f"[{len(self.raw_dataframe)}/{self.data_array_dim[0]}] - Data points done!")
            with self._progress_lock:
                self.raw_dataframe_progress = self._processed_rows / self.data_array_dim[0]
            if status:
                self.dataframe_loop_stop.set()
                print("Finished building raw dataloop!!")
            time.sleep(self.scanning_interval)

    def get_raw_dataframe_progress(self) -> float:
        with self._progress_lock:
            return self.raw_dataframe_progress
        
      
    def filter_by_time_stamps(self, cutoff_rate: float = 1, t_lim: np.ndarray|None = None) -> pd.DataFrame:
        if t_lim is None or len(t_lim)==0:
            return self.get_dataframe()
        df = self.get_dataframe()  # ONE copy, not 200
        if df.empty:
            return df
        t_begin = df["t_begin"].to_numpy()
        t_end   = df["t_end"].to_numpy()
        # Build a boolean mask across all rows at once
        mask = np.zeros(len(df), dtype=bool)
        
        for i in range(len(t_lim)):
            t_krit_min = t_lim[i, 0]
            t_krit_max = t_lim[i, 1]
            # Vectorized cutoff calculation
            measurement_time = t_end - t_begin
            
            # Rows completely outside the window get cutoff=0
            outside = (t_end < t_krit_min) | (t_begin > t_krit_max)
            
            # For rows inside: overlap = min(t_end, t_krit_max) - max(t_begin, t_krit_min)
            overlap = np.minimum(t_end, t_krit_max) - np.maximum(t_begin, t_krit_min)
            overlap = np.where(outside, 0.0, overlap)
            
            cutoff = np.where(measurement_time > 0, overlap / measurement_time, 0.0)
            mask |= (cutoff >= cutoff_rate)
    
        return df[mask].reset_index(drop=True)
        """ old code - takes too long
        else:
            filtered_rows = []
            for i in range(len(t_lim)):
                t_krit_min = t_lim[i,0]
                t_krit_max = t_lim[i,1]
                for _, row in self.get_dataframe().iterrows():
                    t_min = row["t_begin"]
                    t_max = row["t_end"]
                    cutoff = self.calculate_cutoff_rate(t_krit_min, t_krit_max, t_min, t_max)
                    if cutoff >= cutoff_rate:
                        filtered_rows.append(row.to_dict())
            filtered_dataframe = pd.DataFrame(filtered_rows)
            return filtered_dataframe.copy()
        """      
    # This calculates the proportion in the time frame    
    def calculate_cutoff_rate(self, t_krit_min:float, t_krit_max:float, t_min:float, t_max: float) -> float:
            measurement_time = t_max-t_min
            if t_max < t_krit_min or t_min > t_krit_max:
                cutoff = 0
            else:
                time_list=sorted([t_krit_min, t_krit_max, t_min,t_max])
                time_in_range = time_list[2] - time_list[1]
                cutoff = time_in_range/measurement_time
            return cutoff
    
        #Transform data to be sorted after x_points and filter with time_stamps
    def evaluate_qs_data(self, cutoff_rate: float = 1, time_array: np.ndarray = []) -> dict:
        filtered_data = self.filter_by_time_stamps(cutoff_rate=cutoff_rate, t_lim=time_array)
        return_data = {
            "x_data" : 0,
            "reference" : 0,
            "value" : 0,
            "contrast" : 0,
            "time_measure":0,
            "time_save":0,
            "reference_mean" : 0,
            "value_mean" :0,
            "contrast_mean" :0,
            "time_measure_mean":0,
            "time_save_mean":0,
            "reference_std":0,
            "value_std":0,
            "contrast_std":0,
            "time_measure_std":0,
            "time_save_std":0,
            "reference_length":0,
            "value_length":0,
            "contrast_length":0,
            "time_measure_length":0,
            "time_save_length":0,
            "length" : 0,
            "x_unit" : self.x_unit,
            "x_label" : self.x_type,
            "experiment": self.experiment_type
            }
        if filtered_data.empty:
            print("filtered Data are empty")
            return return_data
        if self.averaging_mode == "sum":
            x_data, y_ref, y_val, y_contr, t_begin, t_end = self.evaluate_sum_mode(filtered_data)
        elif self.averaging_mode == "spread":
            x_data, y_ref, y_val, y_contr,t_begin,t_end = self.evaluate_spread_mode(filtered_data)
            
        t_measure = [np.subtract(t_end[i], t_begin[i]) for i in range(len(t_begin))]
        t_save = []
        for j in range(len(t_begin)):
            # sort indices by t_begin
            sorted_idx = np.argsort(t_begin[j])
            t_b = np.array(t_begin[j])[sorted_idx]
            t_e = np.array(t_end[j])[sorted_idx]
            # compute gaps
            gaps = [t_b[i] - t_e[i-1] for i in range(1, len(t_b))]
            t_save.append(gaps)
        
        y_contr_mean = [np.mean(datapoint) for datapoint in y_contr]
        y_ref_mean =  [np.mean(datapoint) for datapoint in y_ref]
        y_val_mean =  [np.mean(datapoint) for datapoint in y_val]
        t_measure_mean = [np.mean(datapoint) for datapoint in t_measure]
        t_save_mean = [np.mean(datapoint) for datapoint in t_save]
       
        
        y_contr_std = [np.std(datapoint)/np.sqrt(len(datapoint)) for datapoint in y_contr] 
        y_ref_std =  [np.std(datapoint)/np.sqrt(len(datapoint)) for datapoint in y_ref]
        y_val_std =  [np.std(datapoint)/np.sqrt(len(datapoint)) for datapoint in y_val]
        t_measure_std = [np.std(datapoint)/np.sqrt(len(datapoint)) for datapoint in t_measure]
        t_save_std = [np.std(datapoint)/np.sqrt(len(datapoint)) for datapoint in t_save]
        
        y_contr_len = [len(datapoint) for datapoint in y_contr] 
        y_ref_len =  [len(datapoint) for datapoint in y_ref]
        y_val_len =  [len(datapoint) for datapoint in y_val]
        t_measure_len = [len(datapoint) for datapoint in t_measure] 
        t_save_len = [len(datapoint) for datapoint in t_save] 
        
        length = min([y_contr_len, y_ref_len, y_val_len])
        
        return_data["x_data"] = x_data
        return_data["reference"] = y_ref
        return_data["value"] = y_val
        return_data["contrast"] = y_contr
        return_data["time_measure"] = t_measure
        return_data["time_save"] = t_save
        return_data["reference_mean"] = y_ref_mean 
        return_data["value_mean"] = y_val_mean
        return_data["contrast_mean"] = y_contr_mean
        return_data["time_measure_mean"] = t_measure_mean
        return_data["time_save_mean"] = t_save_mean
        return_data["reference_std"] = y_ref_std
        return_data["value_std"] = y_val_std
        return_data["contrast_std"] = y_contr_std
        return_data["time_measure_std"] = t_measure_std
        return_data["time_save_std"] = t_save_std
        return_data["reference_length"] = y_ref_len
        return_data["value_length"]= y_val_len
        return_data["contrast_length"] = y_contr_len
        return_data["time_measure_length"] =t_measure_len
        return_data["time_save_length"] = t_save_len
        return_data["length"] = length
        return return_data
   
    def evaluate_sum_mode(self, filtered_data) -> pd.DataFrame():
        x_data = {}
        for key, t_begin, t_end, ch0, ch1, contr in zip(
            filtered_data[self.x_type],
            filtered_data["t_begin"],
            filtered_data["t_end"],
            filtered_data["channel_0"],
            filtered_data["channel_1"],
            filtered_data["contrast"],
        ):
    
            if key not in x_data:
                x_data[key] = [[],[],[], [], []]
    
            x_data[key][0].append(ch0)
            x_data[key][1].append(ch1)
            x_data[key][2].append(contr)
            # ensure iterable
            t_begins = t_begin if isinstance(t_begin, (list, np.ndarray)) else [t_begin]
            t_ends   = t_end   if isinstance(t_end, (list, np.ndarray)) else [t_end]
            x_data[key][3].extend(t_begins)
            x_data[key][4].extend(t_ends)
            
        keys = list(x_data.keys())
        values = list(x_data.values())
    
        y_ref   = [v[0] for v in values]
        y_val   = [v[1] for v in values]
        y_contr = [v[2] for v in values]
        t_begin = [v[3] for v in values]
        t_end = [v[4] for v in values]
        return keys, y_ref, y_val, y_contr, t_begin, t_end
    
    def evaluate_spread_mode(self, filtered_data) -> pd.DataFrame():
        x_data = dict()
        for x_axis_values, ch0_row, ch1_row, contrast_row, t_begin, t_end in zip(
            filtered_data[self.x_type],
            filtered_data["channel_0"],
            filtered_data["channel_1"],
            filtered_data["contrast"],
            filtered_data["t_begin"],
            filtered_data["t_end"],
        ):
            for x, ch0, ch1, contr,t_0,t_1 in zip(
                x_axis_values,
                ch0_row,
                ch1_row,
                contrast_row,
                [t_begin] * len(x_axis_values),
                [t_end] * len(x_axis_values)
            ):
                if x not in x_data:
                    x_data[x] = [[], [], [],[],[]]
    
                x_data[x][0].append(ch0)
                x_data[x][1].append(ch1)
                x_data[x][2].append(contr)
                x_data[x][3].append(t_0)
                x_data[x][4].append(t_1)
    
        # convert to array
        y_data = np.array(list(x_data.values()), dtype=object)
        #print(y_data)
        # extract aligned columns
        y_ref   = y_data[:,0]
        y_val   = y_data[:,1]
        y_contr = y_data[:,2]
        t_begin = y_data[:,3]
        t_end = y_data[:,4]
        return list(x_data.keys()), list(y_ref), list(y_val), list(y_contr), list(t_begin), list(t_end)





