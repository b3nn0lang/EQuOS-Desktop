# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:06:37 2026

@author: Benno Dauer
"""

import ctypes
from time import time
import os
import threading
import pandas as pd
try:
    import zstandard  # force early load before pythonnet
except ImportError:
    pass
import numpy as np
import matplotlib.pyplot as plt
import pypalmsens as ps
import nest_asyncio
nest_asyncio.apply()
class Potentiostat():
    def __init__(self, folder_path):
        ps.settings.General.power_frequency = 50
        self.folder_path = folder_path
        self.start_time_lock = threading.RLock()
        self.callback_data_lock = threading.RLock()
        self.result_lock = threading.RLock()
        self.measurement_start_time = []
        self.callback_data = { "Potential": [], "Current": []}
        self.measurement_results = [[], pd.DataFrame()]
        self.avg_per_point= 0
        self.n_scans = 0
        self.max_points=0
        self.measurement_name = ""
        self.method=None
        self.stop_measurement = threading.Event()
        self.potentiostat_disconnected = threading.Event()
        self.potentiostat_disconnected.set()
        #create cv method
    def cv_measurement(self, begin_potential:float, end_potential:float, step_potential:float, step_rate:float) -> ps.CyclicVoltammetry:
        
        method = ps.CyclicVoltammetry(begin_potential=begin_potential, vertex1_potential = begin_potential,
                                      vertex2_potential=end_potential, n_scans=1,step_potential=step_potential,scanrate=step_rate)
        self.max_points = (abs(begin_potential-end_potential)/step_potential)*2
        self.measurement_name = "Cyclic Voltammetry"
        self.method=method
        return method
    def swv_measurement(self, begin_potential:float, end_potential:float,step_potential:float,frequency:float, amplitude:float) -> ps.SquareWaveVoltammetry:
        method = ps.SquareWaveVoltammetry(begin_potential=begin_potential, end_potential=end_potential,
                                          step_potential=step_potential, frequency=frequency,amplitude=amplitude,
                                          record_forward_and_reverse_currents=False)
        self.max_points = abs(begin_potential-end_potential)/step_potential
        self.measurement_name="Square Wave Voltammetry"
        self.method=method
        return method
    def lsv_measurement(self,begin_potential:float, end_potential:float, step_potential:float, scanrate:float) ->ps.LinearSweepVoltammetry:
        method = ps.LinearSweepVoltammetry(begin_potential=begin_potential, end_potential=end_potential, step_potential=step_potential, scanrate=scanrate)
        self.max_points = (abs(begin_potential-end_potential)/step_potential)
        self.measurement_name = "Linear Sweep Voltammetry"
        self.method=method
        return method
    def get_callback_data(self) -> dict:
        with self.callback_data_lock:
            return self.callback_data.copy()
    def set_callback_data(self,data: list):
        with self.callback_data_lock:
            self.callback_data = data
    def get_measurement_start_time(self) -> list:
        with self.start_time_lock:
            return self.measurement_start_time.copy()
    def set_measurement_start_time(self, time:list):
        with self.start_time_lock:
            self.measurement_start_time = time
    def set_measurement_results(self, result):
        with self.result_lock:
            self.measurement_results = result
    def get_measurement_results(self):
        with self.result_lock:
            return self.measurement_results.copy()
    def get_measurement_count(self) -> int:
        with self.result_lock:
            return len(self.measurement_results[0])
    def get_measurement_integrals(self) -> list[float]:
        with self.result_lock:
            measurements = self.measurement_results[0]
            integrals = []
            for measurement in measurements:
                if hasattr(measurement, "dataset"):
                    integrals.append(float(np.sum(np.abs(measurement.dataset.arrays(type='Current')))))
                elif isinstance(measurement, dict) and "current_integral" in measurement:
                    integrals.append(float(measurement["current_integral"]))
            return integrals
    def get_measurement_plots(self) ->list:
        with self.result_lock:
            measurements = self.measurement_results[0]
        plots = []
        for measurement in measurements:
            figure = measurement.curves[0].plot()
            plots.append(figure)
        return plots
    def append_measurement_frame(self, results_df: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
        if results_df.empty:
            return new_rows.reset_index(drop=True)
        return pd.concat([results_df, new_rows], ignore_index=True, copy=False)
        #callback plots the data points not saved and calculates the time of the first index point
    def callback(self,data: list()):
        start_time = time()
        if self.get_measurement_start_time() == []:
            self.set_measurement_start_time([start_time, data.index])
        data_dict = self.get_callback_data()
        start = data.start  # index where new data begins
        if not data_dict:
            data_dict = {
                "Potential": [],
                "Current": []
            }

        # append new data
        n = min(len(data.x_array[start:]), len(data.y_array[start:]))
        data_dict["Potential"].extend(data.x_array[start:start+n])
        data_dict["Current"].extend(data.y_array[start:start+n])
    
        self.set_callback_data(data_dict)
        
        #this method takes
    def evaluated_data(self) -> pd.DataFrame:
        raw_data = self.get_measurement_results()
        raw_df = raw_data[1]
        if raw_df.empty:
            return None
        else:
            rounded_potential = raw_df["Potential"].round(5)
            eval_data = raw_df.assign(Potential=rounded_potential).groupby(["direction","Potential"],as_index=False).agg(list)
            eval_data["Current_mean"] = eval_data["Current"].apply(np.mean)
            #eval_data["Charge_mean"] = eval_data["Charge"].apply(np.mean)
            eval_data["Current_std"] = eval_data["Current"].apply(np.std)
           # eval_data["Charge_std"] = eval_data["Charge"].apply(np.std)
        return eval_data

    def measure_n_times(self, file_name, method, n_scans):  
        results_df = pd.DataFrame()
        measurement_results= []
        session_measurements = []
        ctypes.windll.winmm.timeBeginPeriod(1)  
        self.stop_measurement.clear()
        self.potentiostat_disconnected.clear()
        self.set_measurement_results([[], pd.DataFrame()])
        with ps.connect() as manager:
                for i in range(n_scans):
                        if self.stop_measurement.is_set():
                            break
                        self.set_measurement_start_time([])
                        self.set_callback_data( {"Potential": [],
                         "Current": []})
                        print("Starting EC measurement")
                        measurement = manager.measure(method, callback=self.callback)
                        print("EC measurement finished")
                        session_measurements.append(measurement)
                        measurement_results.append({
                            "current_integral": float(np.sum(np.abs(measurement.dataset.arrays(type='Current'))))
                        })
                        new_df = pd.DataFrame(measurement.dataset.to_dict())
                        corrected_df = self.correct_time_stamps(self.get_measurement_start_time(), new_df)
                        results_df = self.append_measurement_frame(results_df, corrected_df)
                        self.set_measurement_results([measurement_results, results_df])
                        self.save_measurement_to_file(session_measurements, file_name)
                        self.save_df_to_csv(results_df, file_name)
                manager.disconnect()
                self.potentiostat_disconnected.set()
        print("Potentiostat disonnected")
        self.set_callback_data( {"Potential": [],
         "Current": []})    
        self.set_measurement_start_time([])        
        ctypes.windll.winmm.timeEndPeriod(1)
        print("EC Experiment finished")
        
    def measure_until_done(self,file_name,method):
        results_df = pd.DataFrame()
        measurement_results= []
        session_measurements = []
        ctypes.windll.winmm.timeBeginPeriod(1)  
        self.stop_measurement.clear()
        self.potentiostat_disconnected.clear()
        self.set_measurement_results([[], pd.DataFrame()])
        with ps.connect() as manager:
            while not self.stop_measurement.is_set():
                self.set_measurement_start_time([])
                self.set_callback_data( {"Potential": [],
                 "Current": []})
                print("Starting EC measurement")
                measurement = manager.measure(method, callback=self.callback)
                print("EC measurement finished")
                session_measurements.append(measurement)
                measurement_results.append({
                    "current_integral": float(np.sum(np.abs(measurement.dataset.arrays(type='Current'))))
                })
                new_df = pd.DataFrame(measurement.dataset.to_dict())
                corrected_df = self.correct_time_stamps(self.get_measurement_start_time(), new_df)
                results_df = self.append_measurement_frame(results_df, corrected_df)
                self.set_measurement_results([measurement_results, results_df])
                self.save_measurement_to_file(session_measurements, file_name)
                self.save_df_to_csv(results_df, file_name)
            manager.disconnect()    
            self.potentiostat_disconnected.set()
        print("Potentiostat disonnected")
        self.set_callback_data( {"Potential": [],
         "Current": []})    
        self.set_measurement_start_time([])        
        ctypes.windll.winmm.timeEndPeriod(1)
        print("EC Experiment finished")
    def correct_time_stamps(self,start_time:list, df:pd.DataFrame) -> pd.DataFrame:
        time_to_correct = df["Time"][start_time[1]]
        true_start_time = start_time[0] - time_to_correct
        df["Time"] += true_start_time
        df["Time_begin"] = [df["Time"].iloc[0]] + [
        (df["Time"].iloc[i] + df["Time"].iloc[i-1]) / 2 
        for i in range(1, len(df["Time"]))
        ]
        df["Time_end"] = [(df["Time"].iloc[i]+df["Time"].iloc[i+1])/2 for i in range(len(df["Time"])-1)] + [df["Time"].iloc[-1]]
        df["direction"] = [1 if df["Potential"].iloc[1] > df["Potential"].iloc[0] else -1] + [
            1 if df["Potential"].iloc[i+1] > df["Potential"].iloc[i-1] else -1
            for i in range(1, len(df["Potential"]) - 1)
        ] + [1 if df["Potential"].iloc[-1] > df["Potential"].iloc[-2] else -1]
        return df
        pass
    def save_measurement_to_file(self,measurements,file_name):
        ps.save_session_file(os.path.join(self.folder_path, f"{file_name}.pssession"), measurements)
    def save_df_to_csv(self, df : pd.DataFrame, file_name) -> None:
        df.to_csv(os.path.join(self.folder_path, f"{file_name}.csv"), index=False)
    def load_measurement_from_file(self, file_name):
        self.set_measurement_results([self.load_pssession_file(file_name),self.load_csv_file(file_name)])
        self.n_scans = len(self.get_measurement_results()[0])
        self.method=self.get_measurement_results()[0][0].method
        self.avg_per_point=1 #Dummy value
        settings = self.method.to_settings()  # returns the typed BaseTechnique dataclass
        if self.method.id == "cv":
            self.measurement_name = "Cyclic Voltammetry"
            self.max_points = int(
                abs(settings.vertex1_potential - settings.vertex2_potential)
                / settings.step_potential
            ) * 2
        
        elif self.method.id == "swv":
            self.measurement_name = "Square Wave Voltammetry"
            self.max_points = int(abs(
                settings.begin_potential - settings.end_potential
            ) / settings.step_potential)
        elif self.method.id =="lsv":
            self.measurement_name = "Linear Sweep Voltammetry"
            self.max_points = int(abs(
                settings.begin_potential - settings.end_potential
            ) / settings.step_potential)
    def discover_devices(self):
        return ps.discover()
    def load_csv_file(self,file_name):
        df = pd.read_csv(os.path.join(self.folder_path, f"{file_name}.csv"))
        return df
    def load_pssession_file(self, file_name):
        measurements = ps.load_session_file(os.path.join(self.folder_path, f"{file_name}.pssession"))
        return measurements
    def release_resources(self):
        self.stop_measurement.set()
        self.potentiostat_disconnected.set()
        self.set_callback_data({"Potential": [], "Current": []})
        self.set_measurement_start_time([])
        self.method = None
        self.measurement_name = ""
        self.n_scans = 0
        self.max_points = 0
        self.avg_per_point = 0
        self.set_measurement_results([[], pd.DataFrame()])
    def start_n_measurements(self,file_name,method,n_scans):
        thread = threading.Thread(target=self.measure_n_times, args=(file_name, method, n_scans),daemon=True)
        self.n_scans = n_scans
        thread.start()
        print("EC Experiment started")
    def start_measurement_until_stop(self,file_name,method,avg_per_point):
        thread = threading.Thread(target=self.measure_until_done, args=(file_name,method),daemon=True)
        self.avg_per_point = avg_per_point
        thread.start()
        print("EC Experiment started")
    
  


