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
        self._summary_lock = threading.RLock()
        self._progress_lock = threading.RLock()
        self.dataframe_loop_stop = threading.Event()
        self.raw_dataframe_progress = 0
        self.scanning_interval = 0.5
        self.qupyt_is_dead_time = 3
        self.qupyt_param_creation_time = 10
        self.temp_path = "C:/Users/ge54vec/.qupyt/temp"
        self.file_path = file_path
        self.file_name = file_name
        self.params = None
        self.experiment_type = None
        
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
        if self.params is None:
            raise FileNotFoundError(
                f"{self.file_path}/{self.file_name}.yaml has not been found after waiting "
                f"{self.qupyt_param_creation_time:.1f} s"
            )
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
            print(f"{self.experiment_type or 'unknown experiment'}")
            raise KeyError(f"One of the parameters in {self.file_name}.yaml has not been defined. Have fun looking: {e}")
            

        self.data_array_dim = self.build_data_file_dimension()
        self.points_per_row = self.number_measurements // self.reference_channels
        self._x_values_array = np.asarray(self.x_values, dtype=float)
        self._active_dat_file = None
        self._processed_rows = 0
        self._live_summary = self._init_live_summary()

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

    def build_dataframe(self, dat_file: str, filled_rows: int) -> None:
        if filled_rows < self._processed_rows:
            raise EOFError("Dataframe longer than raw Data")
        if filled_rows > self._processed_rows:
            self._update_live_summary(dat_file, self._processed_rows, filled_rows)
        self._active_dat_file = dat_file
        self._processed_rows = filled_rows

    def _empty_result(self) -> dict:
        return {
            "x_data" : [],
            "reference" : [],
            "value" : [],
            "contrast" : [],
            "time_measure": [],
            "time_save": [],
            "reference_mean" : [],
            "value_mean" : [],
            "contrast_mean" : [],
            "time_measure_mean": [],
            "time_save_mean": [],
            "reference_std": [],
            "value_std": [],
            "contrast_std": [],
            "time_measure_std": [],
            "time_save_std": [],
            "reference_length": [],
            "value_length": [],
            "contrast_length": [],
            "time_measure_length": [],
            "time_save_length": [],
            "length" : 0,
            "x_unit" : self.x_unit,
            "x_label" : self.x_type,
            "experiment": self.experiment_type
            }

    def _init_live_summary(self) -> dict:
        n_x = len(self._x_values_array)
        if self.averaging_mode == "spread":
            return {
                "count": 0,
                "sum_ref": np.zeros(n_x, dtype=float),
                "sum_val": np.zeros(n_x, dtype=float),
                "sum_contr": np.zeros(n_x, dtype=float),
                "sumsq_ref": np.zeros(n_x, dtype=float),
                "sumsq_val": np.zeros(n_x, dtype=float),
                "sumsq_contr": np.zeros(n_x, dtype=float),
                "time_measure_sum": 0.0,
                "time_measure_sumsq": 0.0,
                "time_measure_count": 0,
                "time_save_sum": 0.0,
                "time_save_sumsq": 0.0,
                "time_save_count": 0,
                "last_end": None,
            }
        return {
            "count": np.zeros(n_x, dtype=int),
            "sum_ref": np.zeros(n_x, dtype=float),
            "sum_val": np.zeros(n_x, dtype=float),
            "sum_contr": np.zeros(n_x, dtype=float),
            "sumsq_ref": np.zeros(n_x, dtype=float),
            "sumsq_val": np.zeros(n_x, dtype=float),
            "sumsq_contr": np.zeros(n_x, dtype=float),
            "time_measure_sum": np.zeros(n_x, dtype=float),
            "time_measure_sumsq": np.zeros(n_x, dtype=float),
            "time_measure_count": np.zeros(n_x, dtype=int),
            "time_save_sum": np.zeros(n_x, dtype=float),
            "time_save_sumsq": np.zeros(n_x, dtype=float),
            "time_save_count": np.zeros(n_x, dtype=int),
            "last_end": np.full(n_x, np.nan, dtype=float),
        }

    def _update_live_summary(self, dat_file: str, row_start: int, row_stop: int) -> None:
        mm = np.memmap(dat_file, dtype=float, mode="r", shape=self.data_array_dim)
        try:
            raw_rows = np.asarray(mm[row_start:row_stop])
        finally:
            del mm

        if len(raw_rows) == 0:
            return

        t_begin = raw_rows[:, 0]
        t_end = raw_rows[:, 1]
        signal_rows = raw_rows[:, 2:].reshape(-1, self.points_per_row, self.reference_channels)
        channel_0 = signal_rows[:, :, 0]
        channel_1 = signal_rows[:, :, 1]
        contrast = np.divide(channel_1, channel_0)

        with self._summary_lock:
            summary = self._live_summary
            if self.averaging_mode == "spread":
                summary["count"] += len(raw_rows)
                summary["sum_ref"] += np.sum(channel_0, axis=0)
                summary["sum_val"] += np.sum(channel_1, axis=0)
                summary["sum_contr"] += np.sum(contrast, axis=0)
                summary["sumsq_ref"] += np.sum(channel_0 ** 2, axis=0)
                summary["sumsq_val"] += np.sum(channel_1 ** 2, axis=0)
                summary["sumsq_contr"] += np.sum(contrast ** 2, axis=0)
                durations = t_end - t_begin
                summary["time_measure_sum"] += float(np.sum(durations))
                summary["time_measure_sumsq"] += float(np.sum(durations ** 2))
                summary["time_measure_count"] += len(durations)

                gap_list = []
                if summary["last_end"] is not None:
                    gap_list.append(float(t_begin[0] - summary["last_end"]))
                if len(t_begin) > 1:
                    gap_list.extend((t_begin[1:] - t_end[:-1]).tolist())
                if gap_list:
                    gaps = np.asarray(gap_list, dtype=float)
                    summary["time_save_sum"] += float(np.sum(gaps))
                    summary["time_save_sumsq"] += float(np.sum(gaps ** 2))
                    summary["time_save_count"] += len(gaps)
                summary["last_end"] = float(t_end[-1])
            else:
                row_x_idx = np.arange(row_start, row_stop) % len(self._x_values_array)
                durations = t_end - t_begin
                for local_i, x_idx in enumerate(row_x_idx):
                    summary["count"][x_idx] += 1
                    ch0 = channel_0[local_i]
                    ch1 = channel_1[local_i]
                    contr = contrast[local_i]
                    summary["sum_ref"][x_idx] += float(np.sum(ch0))
                    summary["sum_val"][x_idx] += float(np.sum(ch1))
                    summary["sum_contr"][x_idx] += float(np.sum(contr))
                    summary["sumsq_ref"][x_idx] += float(np.sum(ch0 ** 2))
                    summary["sumsq_val"][x_idx] += float(np.sum(ch1 ** 2))
                    summary["sumsq_contr"][x_idx] += float(np.sum(contr ** 2))
                    summary["time_measure_sum"][x_idx] += float(durations[local_i])
                    summary["time_measure_sumsq"][x_idx] += float(durations[local_i] ** 2)
                    summary["time_measure_count"][x_idx] += 1
                    if not np.isnan(summary["last_end"][x_idx]):
                        gap = float(t_begin[local_i] - summary["last_end"][x_idx])
                        summary["time_save_sum"][x_idx] += gap
                        summary["time_save_sumsq"][x_idx] += gap ** 2
                        summary["time_save_count"][x_idx] += 1
                    summary["last_end"][x_idx] = float(t_end[local_i])

    def _live_summary_snapshot(self) -> dict:
        result = self._empty_result()
        with self._summary_lock:
            summary = self._live_summary
            if self.averaging_mode == "spread":
                n_rows = int(summary["count"])
                if n_rows == 0:
                    return result
                n_points = len(self._x_values_array)
                denom = max(n_rows, 1)
                result["x_data"] = self._x_values_array.tolist()
                result["reference_mean"] = (summary["sum_ref"] / denom).tolist()
                result["value_mean"] = (summary["sum_val"] / denom).tolist()
                result["contrast_mean"] = (summary["sum_contr"] / denom).tolist()
                result["reference_std"] = np.sqrt(np.maximum(summary["sumsq_ref"] / denom - (summary["sum_ref"] / denom) ** 2, 0.0)).tolist()
                result["value_std"] = np.sqrt(np.maximum(summary["sumsq_val"] / denom - (summary["sum_val"] / denom) ** 2, 0.0)).tolist()
                result["contrast_std"] = np.sqrt(np.maximum(summary["sumsq_contr"] / denom - (summary["sum_contr"] / denom) ** 2, 0.0)).tolist()
                result["reference_std"] = (np.asarray(result["reference_std"]) / np.sqrt(denom)).tolist()
                result["value_std"] = (np.asarray(result["value_std"]) / np.sqrt(denom)).tolist()
                result["contrast_std"] = (np.asarray(result["contrast_std"]) / np.sqrt(denom)).tolist()
                tm_count = int(summary["time_measure_count"])
                ts_count = int(summary["time_save_count"])
                tm_mean = summary["time_measure_sum"] / tm_count if tm_count else 0.0
                ts_mean = summary["time_save_sum"] / ts_count if ts_count else 0.0
                tm_std = np.sqrt(max(summary["time_measure_sumsq"] / tm_count - tm_mean ** 2, 0.0)) / np.sqrt(tm_count) if tm_count else 0.0
                ts_std = np.sqrt(max(summary["time_save_sumsq"] / ts_count - ts_mean ** 2, 0.0)) / np.sqrt(ts_count) if ts_count else 0.0
                result["time_measure_mean"] = [float(tm_mean)] * n_points
                result["time_save_mean"] = [float(ts_mean)] * n_points
                result["time_measure_std"] = [float(tm_std)] * n_points
                result["time_save_std"] = [float(ts_std)] * n_points
                result["reference_length"] = [n_rows] * n_points
                result["value_length"] = [n_rows] * n_points
                result["contrast_length"] = [n_rows] * n_points
                result["time_measure_length"] = [tm_count] * n_points
                result["time_save_length"] = [ts_count] * n_points
                result["length"] = n_rows
                return result

            counts = summary["count"]
            valid = counts > 0
            if not np.any(valid):
                return result
            x_idx = np.flatnonzero(valid)
            elem_counts = counts[valid] * self.points_per_row
            result["x_data"] = self._x_values_array[valid].tolist()
            result["reference_mean"] = (summary["sum_ref"][valid] / elem_counts).tolist()
            result["value_mean"] = (summary["sum_val"][valid] / elem_counts).tolist()
            result["contrast_mean"] = (summary["sum_contr"][valid] / elem_counts).tolist()
            ref_var = np.maximum(summary["sumsq_ref"][valid] / elem_counts - (summary["sum_ref"][valid] / elem_counts) ** 2, 0.0)
            val_var = np.maximum(summary["sumsq_val"][valid] / elem_counts - (summary["sum_val"][valid] / elem_counts) ** 2, 0.0)
            contr_var = np.maximum(summary["sumsq_contr"][valid] / elem_counts - (summary["sum_contr"][valid] / elem_counts) ** 2, 0.0)
            result["reference_std"] = (np.sqrt(ref_var) / np.sqrt(counts[valid])).tolist()
            result["value_std"] = (np.sqrt(val_var) / np.sqrt(counts[valid])).tolist()
            result["contrast_std"] = (np.sqrt(contr_var) / np.sqrt(counts[valid])).tolist()
            tm_count = summary["time_measure_count"][valid]
            ts_count = summary["time_save_count"][valid]
            tm_mean = np.divide(summary["time_measure_sum"][valid], tm_count, out=np.zeros_like(summary["time_measure_sum"][valid]), where=tm_count > 0)
            ts_mean = np.divide(summary["time_save_sum"][valid], ts_count, out=np.zeros_like(summary["time_save_sum"][valid]), where=ts_count > 0)
            tm_var = np.divide(summary["time_measure_sumsq"][valid], tm_count, out=np.zeros_like(summary["time_measure_sumsq"][valid]), where=tm_count > 0) - tm_mean ** 2
            ts_var = np.divide(summary["time_save_sumsq"][valid], ts_count, out=np.zeros_like(summary["time_save_sumsq"][valid]), where=ts_count > 0) - ts_mean ** 2
            result["time_measure_mean"] = tm_mean.tolist()
            result["time_save_mean"] = ts_mean.tolist()
            result["time_measure_std"] = (np.sqrt(np.maximum(tm_var, 0.0)) / np.sqrt(np.maximum(tm_count, 1))).tolist()
            result["time_save_std"] = (np.sqrt(np.maximum(ts_var, 0.0)) / np.sqrt(np.maximum(ts_count, 1))).tolist()
            result["reference_length"] = counts[valid].tolist()
            result["value_length"] = counts[valid].tolist()
            result["contrast_length"] = counts[valid].tolist()
            result["time_measure_length"] = tm_count.tolist()
            result["time_save_length"] = ts_count.tolist()
            result["length"] = int(np.min(counts[valid]))
            return result

    def _load_raw_data(self, row_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._active_dat_file is None or self._processed_rows == 0:
            empty_signal = np.empty((0, self.points_per_row, self.reference_channels), dtype=float)
            return np.array([], dtype=float), np.array([], dtype=float), empty_signal

        mm = np.memmap(self._active_dat_file, dtype=float, mode="r", shape=self.data_array_dim)
        try:
            if row_indices is None:
                raw_rows = mm[:self._processed_rows]
            else:
                raw_rows = mm[row_indices]
            t_begin = np.array(raw_rows[:, 0], copy=True)
            t_end = np.array(raw_rows[:, 1], copy=True)
            signal_rows = np.array(raw_rows[:, 2:], copy=True).reshape(-1, self.points_per_row, self.reference_channels)
        finally:
            del mm
        return t_begin, t_end, signal_rows

    def get_dataframe(self) -> pd.DataFrame:
        t_begin, t_end, signal_rows = self._load_raw_data()
        if len(t_begin) == 0:
            return pd.DataFrame()

        channel_dic = {
            ch: [row[:, ch].copy() for row in signal_rows]
            for ch in range(self.reference_channels)
        }
        contrast = [np.divide(row[:, 1], row[:, 0]) for row in signal_rows]
        if self.averaging_mode == "spread":
            x_values = [self._x_values_array.tolist() for _ in range(len(t_begin))]
        elif len(self._x_values_array) > 0:
            x_values = self._x_values_array[np.arange(len(t_begin)) % len(self._x_values_array)]
        else:
            x_values = []

        return pd.DataFrame(
            {
                "t_begin": t_begin,
                "t_end": t_end,
                **{f"channel_{key}": val for key, val in channel_dic.items()},
                "contrast": contrast,
                f"{self.x_type}": x_values,
            }
        )
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

    def release_resources(self) -> None:
        self.dataframe_loop_stop.set()
        self._active_dat_file = None
        self._processed_rows = 0
        with self._progress_lock:
            self.raw_dataframe_progress = 0
        with self._summary_lock:
            self._live_summary = self._init_live_summary()
        
      
    def filter_by_time_stamps(self, cutoff_rate: float = 1, t_lim: np.ndarray|None = None) -> np.ndarray:
        if self._processed_rows == 0:
            return np.array([], dtype=int)
        if t_lim is None or len(t_lim)==0:
            return np.arange(self._processed_rows, dtype=int)

        t_begin, t_end, _ = self._load_raw_data()
        # Build a boolean mask across all rows at once
        mask = np.zeros(len(t_begin), dtype=bool)
        
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
    
        return np.flatnonzero(mask)
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
        filtered_indices = self.filter_by_time_stamps(cutoff_rate=cutoff_rate, t_lim=time_array)
        return_data = self._empty_result()
        if time_array is None or len(time_array) == 0:
            return self._live_summary_snapshot()
        if len(filtered_indices) == 0:
            print("filtered Data are empty")
            return return_data
        t_begin_rows, t_end_rows, signal_rows = self._load_raw_data(filtered_indices)
        if self.averaging_mode == "sum":
            summary = self.evaluate_sum_mode(filtered_indices, t_begin_rows, t_end_rows, signal_rows)
        elif self.averaging_mode == "spread":
            summary = self.evaluate_spread_mode(t_begin_rows, t_end_rows, signal_rows)

        return_data["x_data"] = summary["x_data"]
        return_data["reference_mean"] = summary["reference_mean"]
        return_data["value_mean"] = summary["value_mean"]
        return_data["contrast_mean"] = summary["contrast_mean"]
        return_data["time_measure_mean"] = summary["time_measure_mean"]
        return_data["time_save_mean"] = summary["time_save_mean"]
        return_data["reference_std"] = summary["reference_std"]
        return_data["value_std"] = summary["value_std"]
        return_data["contrast_std"] = summary["contrast_std"]
        return_data["time_measure_std"] = summary["time_measure_std"]
        return_data["time_save_std"] = summary["time_save_std"]
        return_data["reference_length"] = summary["reference_length"]
        return_data["value_length"]= summary["value_length"]
        return_data["contrast_length"] = summary["contrast_length"]
        return_data["time_measure_length"] = summary["time_measure_length"]
        return_data["time_save_length"] = summary["time_save_length"]
        return_data["length"] = summary["length"]
        return return_data
   
    def _time_stats(self, t_begin: np.ndarray, t_end: np.ndarray) -> tuple[float, float, int, float, float, int]:
        t_measure = np.subtract(t_end, t_begin)
        t_measure_len = len(t_measure)
        if t_measure_len > 0:
            t_measure_mean = float(np.mean(t_measure))
            t_measure_std = float(np.std(t_measure) / np.sqrt(t_measure_len))
        else:
            t_measure_mean = 0.0
            t_measure_std = 0.0

        if len(t_begin) > 1:
            sorted_idx = np.argsort(t_begin)
            t_b = t_begin[sorted_idx]
            t_e = t_end[sorted_idx]
            gaps = t_b[1:] - t_e[:-1]
            t_save_len = len(gaps)
            t_save_mean = float(np.mean(gaps)) if t_save_len > 0 else 0.0
            t_save_std = float(np.std(gaps) / np.sqrt(t_save_len)) if t_save_len > 0 else 0.0
        else:
            t_save_len = 0
            t_save_mean = 0.0
            t_save_std = 0.0
        return t_measure_mean, t_measure_std, t_measure_len, t_save_mean, t_save_std, t_save_len

    def evaluate_sum_mode(self, filtered_indices, t_begin_rows, t_end_rows, signal_rows) -> dict:
        x_data = []
        reference_mean = []
        value_mean = []
        contrast_mean = []
        time_measure_mean = []
        time_save_mean = []
        reference_std = []
        value_std = []
        contrast_std = []
        time_measure_std = []
        time_save_std = []
        reference_length = []
        value_length = []
        contrast_length = []
        time_measure_length = []
        time_save_length = []
        row_x_values = self._x_values_array[np.asarray(filtered_indices) % len(self._x_values_array)]
        channel_0 = signal_rows[:, :, 0]
        channel_1 = signal_rows[:, :, 1]
        contrast_rows = np.divide(channel_1, channel_0)
        unique_keys = list(dict.fromkeys(row_x_values.tolist()))

        for key in unique_keys:
            mask = row_x_values == key
            ch0_group = channel_0[mask]
            ch1_group = channel_1[mask]
            contrast_group = contrast_rows[mask]
            t_begin_group = t_begin_rows[mask]
            t_end_group = t_end_rows[mask]
            n_group = int(np.count_nonzero(mask))

            x_data.append(float(key))
            reference_mean.append(float(np.mean(ch0_group)))
            value_mean.append(float(np.mean(ch1_group)))
            contrast_mean.append(float(np.mean(contrast_group)))
            reference_std.append(float(np.std(ch0_group) / np.sqrt(n_group)))
            value_std.append(float(np.std(ch1_group) / np.sqrt(n_group)))
            contrast_std.append(float(np.std(contrast_group) / np.sqrt(n_group)))
            reference_length.append(n_group)
            value_length.append(n_group)
            contrast_length.append(n_group)

            t_measure_mean_i, t_measure_std_i, t_measure_len_i, t_save_mean_i, t_save_std_i, t_save_len_i = self._time_stats(
                t_begin_group,
                t_end_group,
            )
            time_measure_mean.append(t_measure_mean_i)
            time_measure_std.append(t_measure_std_i)
            time_measure_length.append(t_measure_len_i)
            time_save_mean.append(t_save_mean_i)
            time_save_std.append(t_save_std_i)
            time_save_length.append(t_save_len_i)

        return {
            "x_data": x_data,
            "reference_mean": reference_mean,
            "value_mean": value_mean,
            "contrast_mean": contrast_mean,
            "time_measure_mean": time_measure_mean,
            "time_save_mean": time_save_mean,
            "reference_std": reference_std,
            "value_std": value_std,
            "contrast_std": contrast_std,
            "time_measure_std": time_measure_std,
            "time_save_std": time_save_std,
            "reference_length": reference_length,
            "value_length": value_length,
            "contrast_length": contrast_length,
            "time_measure_length": time_measure_length,
            "time_save_length": time_save_length,
            "length": min(reference_length) if reference_length else 0,
        }
    
    def evaluate_spread_mode(self, t_begin_rows, t_end_rows, signal_rows) -> dict:
        n_points = min(len(self._x_values_array), signal_rows.shape[1])
        if n_points == 0:
            return {
                "x_data": [],
                "reference_mean": [],
                "value_mean": [],
                "contrast_mean": [],
                "time_measure_mean": [],
                "time_save_mean": [],
                "reference_std": [],
                "value_std": [],
                "contrast_std": [],
                "time_measure_std": [],
                "time_save_std": [],
                "reference_length": [],
                "value_length": [],
                "contrast_length": [],
                "time_measure_length": [],
                "time_save_length": [],
                "length": 0,
            }

        channel_0 = signal_rows[:, :n_points, 0]
        channel_1 = signal_rows[:, :n_points, 1]
        contrast = np.divide(channel_1, channel_0)
        n_rows = signal_rows.shape[0]
        t_measure_mean_i, t_measure_std_i, t_measure_len_i, t_save_mean_i, t_save_std_i, t_save_len_i = self._time_stats(
            t_begin_rows,
            t_end_rows,
        )
        return {
            "x_data": self._x_values_array[:n_points].tolist(),
            "reference_mean": np.mean(channel_0, axis=0).tolist(),
            "value_mean": np.mean(channel_1, axis=0).tolist(),
            "contrast_mean": np.mean(contrast, axis=0).tolist(),
            "time_measure_mean": [t_measure_mean_i] * n_points,
            "time_save_mean": [t_save_mean_i] * n_points,
            "reference_std": (np.std(channel_0, axis=0) / np.sqrt(n_rows)).tolist(),
            "value_std": (np.std(channel_1, axis=0) / np.sqrt(n_rows)).tolist(),
            "contrast_std": (np.std(contrast, axis=0) / np.sqrt(n_rows)).tolist(),
            "time_measure_std": [t_measure_std_i] * n_points,
            "time_save_std": [t_save_std_i] * n_points,
            "reference_length": [n_rows] * n_points,
            "value_length": [n_rows] * n_points,
            "contrast_length": [n_rows] * n_points,
            "time_measure_length": [t_measure_len_i] * n_points,
            "time_save_length": [t_save_len_i] * n_points,
            "length": n_rows,
        }
