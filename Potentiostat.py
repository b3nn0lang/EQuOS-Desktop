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
from tqdm import tqdm
from matplotlib.figure import Figure
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
        self.callback_data = { "Potential": [], "Current": [], "Time": []}
        self.measurement_results = [[], pd.DataFrame()]
        self.avg_per_point= 0
        self.n_scans = 0
        self.max_points=0
        self.no_applied_potential = False
        self.measurement_name = ""
        self.method=None
        self.t1_potential_step = 0.01
        self.ca_sequence_potentials = []
        self.ca_sequence_run_time = None
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
    def ca_measurement(self, potential: float, run_time: float) -> ps.ChronoAmperometry:
        method = ps.ChronoAmperometry(potential=potential, run_time=run_time, interval_time=0.5)
        self.max_points = max(int(np.ceil(run_time / 0.5)), 1)
        self.measurement_name = "Chronoamperometry"
        self.method = method
        return method
    def combined_ca_potentials(self, start_potential: float, length: float, step_potential: float, reverse_to_start: bool) -> list[float]:
        step_potential = abs(float(step_potential))
        if step_potential <= 0:
            raise ValueError("CA step must be larger than 0 V.")
        start_potential = float(start_potential)
        length = float(length)
        distance = abs(length)
        direction = 1 if length >= 0 else -1
        offsets = list(np.arange(0, distance + step_potential * 0.5, step_potential))
        offsets = [offset for offset in offsets if offset <= distance + step_potential * 1e-9]
        if not offsets or not np.isclose(offsets[-1], distance):
            offsets.append(distance)
        potentials = [start_potential + direction * offset for offset in offsets]
        if reverse_to_start and len(potentials) > 1:
            potentials.extend(potentials[-2::-1])
        return [float(np.round(potential, 10)) for potential in potentials]
    def combined_ca_t1_measurement(self, start_potential: float, length: float, step_potential: float, run_time: float, reverse_to_start: bool) -> list[ps.ChronoAmperometry]:
        potentials = self.combined_ca_potentials(start_potential, length, step_potential, reverse_to_start)
        methods = [ps.ChronoAmperometry(potential=potential, run_time=run_time, interval_time=0.5) for potential in potentials]
        self.max_points = max(int(np.ceil(float(run_time) / 0.5)), 1)
        self.measurement_name = "Combined Chronoamperometry"
        self.method = methods[0]
        self.ca_sequence_potentials = potentials
        self.ca_sequence_run_time = float(run_time)
        return methods
    def get_callback_data(self) -> dict:
        with self.callback_data_lock:
            return {
                "Potential": list(self.callback_data.get("Potential", [])),
                "Current": list(self.callback_data.get("Current", [])),
                "Time": list(self.callback_data.get("Time", [])),
            }
    def set_callback_data(self,data: list):
        with self.callback_data_lock:
            self.callback_data = {
                "Potential": list(data.get("Potential", [])),
                "Current": list(data.get("Current", [])),
                "Time": list(data.get("Time", [])),
            }
    def get_measurement_start_time(self) -> list:
        with self.start_time_lock:
            return self.measurement_start_time.copy()
    def set_measurement_start_time(self, time:list):
        with self.start_time_lock:
            self.measurement_start_time = time
    def set_measurement_results(self, result):
        with self.result_lock:
            measurements, results_df = result
            self.measurement_results = [list(measurements), results_df.copy()]
    def get_measurement_results(self):
        with self.result_lock:
            return [list(self.measurement_results[0]), self.measurement_results[1].copy()]
    def get_measurement_count(self) -> int:
        with self.result_lock:
            return len(self.measurement_results[0])

    def normalize_measurement_range(self, measurement_from=1, measurement_to=None) -> tuple[int, int]:
        """Return a clamped, 1-based inclusive measurement range."""
        count = self.get_measurement_count()
        start = max(1, int(measurement_from or 1))
        end = count if measurement_to in (None, 0, "") else min(count, int(measurement_to))
        if count == 0 or start > count or end < start:
            raise ValueError(f"Measurement range {start}-{end} is outside 1-{count}")
        return start, end

    def get_measurement_dataframe(self, measurement_from=1, measurement_to=None) -> pd.DataFrame:
        """Select the contiguous CSV rows belonging to a range of finished scans."""
        with self.result_lock:
            raw_df = self.measurement_results[1].copy()
            count = len(self.measurement_results[0])
        if raw_df.empty or count == 0:
            return raw_df.copy()
        start, end = self.normalize_measurement_range(measurement_from, measurement_to)
        scan_indices = np.array_split(np.arange(len(raw_df)), count)
        selected = np.concatenate(scan_indices[start - 1:end])
        return raw_df.iloc[selected].copy()

    def get_measurement_integrals(self, measurement_from=1, measurement_to=None) -> list[float]:
        with self.result_lock:
            measurements = list(self.measurement_results[0])
        if not measurements:
            return []
        start, end = self.normalize_measurement_range(measurement_from, measurement_to)
        measurements = measurements[start - 1:end]
        with self.result_lock:
            integrals = []
            for measurement in measurements:
                if hasattr(measurement, "dataset"):
                    integrals.append(float(np.sum(np.abs(measurement.dataset.arrays(type='Current')))))
                elif isinstance(measurement, dict) and "current_integral" in measurement:
                    integrals.append(float(measurement["current_integral"]))
            return integrals
    def get_measurement_plots(self, measurement_from=1, measurement_to=None):
        with self.result_lock:
            raw_df = self.measurement_results[1].copy()
            count = len(self.measurement_results[0])
        if raw_df.empty or count == 0:
            return
        start, end = self.normalize_measurement_range(measurement_from, measurement_to)
        scan_indices = np.array_split(np.arange(len(raw_df)), count)
        print("Get EC Plots")
        for measurement_number, indices in tqdm(
            zip(range(start, end + 1), scan_indices[start - 1:end]),
            total=end - start + 1,
        ):
            measurement_df = raw_df.iloc[indices]
            figure = Figure(figsize=(6, 4), constrained_layout=True)
            ax = figure.subplots()
            if getattr(self.method, "id", "") == "ad" and "Time" in measurement_df.columns:
                x_data = measurement_df["Time"] - float(measurement_df["Time"].iloc[0])
                ax.set_xlabel("Time [s]")
            else:
                x_data = measurement_df["Potential"]
                ax.set_xlabel("Potential [V]")
            ax.plot(x_data, measurement_df["Current"], color="tab:blue")
            ax.set_title(f"{self.measurement_name} {measurement_number}")
            ax.set_ylabel("Current [uA]")
            ax.grid(True, alpha=0.3)
            yield figure
    def append_measurement_frame(self, results_df: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
        if results_df.empty:
            return new_rows.reset_index(drop=True)
        return pd.concat([results_df, new_rows], ignore_index=True)
        #callback plots the data points not saved and calculates the time of the first index point
    def ca_callback_time_values(self, existing_points: int, n: int) -> list:
        interval_time = float(getattr(self.method, "interval_time", 0.5))
        return (np.arange(existing_points, existing_points + n, dtype=float) * interval_time).tolist()

    def callback(self,data: list()):
        start_time = time()
        if self.get_measurement_start_time() == []:
            self.set_measurement_start_time([start_time, data.index])
        start = data.start  # index where new data begins

        # append new data
        n = min(len(data.x_array[start:]), len(data.y_array[start:]))
        x_values = list(data.x_array[start:start+n])
        y_values = list(data.y_array[start:start+n])
        with self.callback_data_lock:
            self.callback_data.setdefault("Potential", [])
            self.callback_data.setdefault("Current", [])
            self.callback_data.setdefault("Time", [])
            if self.method is not None and getattr(self.method, "id", "") == "ad":
                self.callback_data["Time"].extend(self.ca_callback_time_values(len(self.callback_data["Current"]), n))
                self.callback_data["Potential"].extend([self.method.potential] * n)
            else:
                self.callback_data["Potential"].extend(x_values)
            self.callback_data["Current"].extend(y_values)
        
        #this method takes
    def evaluated_data(self, measurement_from=1, measurement_to=None) -> pd.DataFrame:
        raw_df = self.get_measurement_dataframe(measurement_from, measurement_to)
        if raw_df.empty:
            return None
        elif self.method is not None and getattr(self.method, "id", "") == "ad":
            eval_data = raw_df.copy()
            eval_data["Relative_time"] = eval_data["Time"] - float(eval_data["Time"].iloc[0])
            return eval_data
        else:
            rounded_potential = raw_df["Potential"].round(5)
            eval_data = raw_df.assign(Potential=rounded_potential).groupby(["direction","Potential"],as_index=False).agg(list)
            eval_data["Current_mean"] = eval_data["Current"].apply(np.mean)
            #eval_data["Charge_mean"] = eval_data["Charge"].apply(np.mean)
            eval_data["Current_std"] = eval_data["Current"].apply(np.std)
           # eval_data["Charge_std"] = eval_data["Charge"].apply(np.std)
        return eval_data

    def start_zero_current_measurement(self, file_name, method):
        """Run the CA-T1 clock without connecting to or commanding any device."""
        if not np.isfinite(method.run_time) or method.run_time <= 0:
            raise ValueError("Run time must be a finite positive number of seconds.")
        self.no_applied_potential = True
        self.measurement_name = "T1 without applied potential (simulated zero current)"
        self.stop_measurement.clear()
        self.potentiostat_disconnected.clear()
        self.set_measurement_results([[], pd.DataFrame()])
        self.set_callback_data({"Potential": [], "Current": [], "Time": []})
        self.set_measurement_start_time([])
        self.n_scans = 1
        threading.Thread(target=self.measure_zero_current, args=(file_name, method), daemon=True).start()

    def measure_zero_current(self, file_name, method):
        from time import monotonic
        start = time()
        clock_start = monotonic()
        elapsed_values = [0.0]
        self.set_measurement_start_time([start, 0])
        try:
            while True:
                elapsed = min(monotonic() - clock_start, float(method.run_time))
                if elapsed > elapsed_values[-1]:
                    elapsed_values.append(elapsed)
                frame = pd.DataFrame({
                    "Time": elapsed_values,
                    "Potential": 0.0,
                    "Current": 0.0,
                })
                frame = self.correct_time_stamps([start, 0], frame)
                frame["No_Applied_Potential"] = 1
                frame["CA_Run_Time"] = float(method.run_time)
                frame["T1_Averages_Per_Datapoint"] = self.avg_per_point
                self.set_measurement_results([[{"current_integral": 0.0}], frame])
                self.save_df_to_csv(frame, file_name)
                if elapsed >= method.run_time or self.stop_measurement.is_set():
                    break
                self.stop_measurement.wait(min(float(method.interval_time), method.run_time - elapsed))
        finally:
            self.set_measurement_start_time([])
            self.potentiostat_disconnected.set()

    def measure_n_times(self, file_name, method, n_scans):  
        results_df = pd.DataFrame()
        measurement_results= []
        session_measurements = []
        ctypes.windll.winmm.timeBeginPeriod(1)  
        try:
            self.stop_measurement.clear()
            self.potentiostat_disconnected.clear()
            self.set_measurement_results([[], pd.DataFrame()])
            with ps.connect() as manager:
                for i in range(n_scans):
                        if self.stop_measurement.is_set():
                            break
                        self.set_measurement_start_time([])
                        self.set_callback_data( {"Potential": [],
                         "Current": [],
                         "Time": []})
                        print("Starting EC measurement")
                        measurement = manager.measure(method, callback=self.callback)
                        print("EC measurement finished")
                        session_measurements.append(measurement)
                        measurement_results.append({
                            "current_integral": float(np.sum(np.abs(measurement.dataset.arrays(type='Current'))))
                        })
                        new_df = pd.DataFrame(measurement.dataset.to_dict())
                        if getattr(method, "id", "") == "ad" and "Potential" not in new_df.columns:
                            new_df["Potential"] = method.potential
                        corrected_df = self.correct_time_stamps(self.get_measurement_start_time(), new_df)
                        results_df = self.append_measurement_frame(results_df, corrected_df)
                        self.set_measurement_results([measurement_results, results_df])
                        self.save_df_to_csv(results_df, file_name)
                manager.disconnect()
                self.potentiostat_disconnected.set()
            if session_measurements:
                self.save_measurement_to_file(session_measurements, file_name)
            print("Potentiostat disonnected")
        finally:
            self.potentiostat_disconnected.set()
            self.set_callback_data( {"Potential": [],
             "Current": [],
             "Time": []})
            self.set_measurement_start_time([])
            ctypes.windll.winmm.timeEndPeriod(1)
            print("EC Experiment finished")
        
    def measure_until_done(self,file_name,method):
        results_df = pd.DataFrame()
        measurement_results= []
        session_measurements = []
        ctypes.windll.winmm.timeBeginPeriod(1)  
        try:
            self.stop_measurement.clear()
            self.potentiostat_disconnected.clear()
            self.set_measurement_results([[], pd.DataFrame()])
            with ps.connect() as manager:
                while not self.stop_measurement.is_set():
                    self.set_measurement_start_time([])
                    self.set_callback_data( {"Potential": [],
                     "Current": [],
                     "Time": []})
                    print("Starting EC measurement")
                    measurement = manager.measure(method, callback=self.callback)
                    print("EC measurement finished")
                    session_measurements.append(measurement)
                    measurement_results.append({
                        "current_integral": float(np.sum(np.abs(measurement.dataset.arrays(type='Current'))))
                    })
                    new_df = pd.DataFrame(measurement.dataset.to_dict())
                    if getattr(method, "id", "") == "ad" and "Potential" not in new_df.columns:
                        new_df["Potential"] = method.potential
                    corrected_df = self.correct_time_stamps(self.get_measurement_start_time(), new_df)
                    results_df = self.append_measurement_frame(results_df, corrected_df)
                    self.set_measurement_results([measurement_results, results_df])
                    self.save_df_to_csv(results_df, file_name)
                manager.disconnect()
                self.potentiostat_disconnected.set()
            if session_measurements:
                self.save_measurement_to_file(session_measurements, file_name)
            print("Potentiostat disonnected")
        finally:
            self.potentiostat_disconnected.set()
            self.set_callback_data( {"Potential": [],
             "Current": [],
             "Time": []})
            self.set_measurement_start_time([])
            ctypes.windll.winmm.timeEndPeriod(1)
            print("EC Experiment finished")
    def measure_method_sequence(self, file_name, methods):
        results_df = pd.DataFrame()
        measurement_results= []
        session_measurements = []
        ctypes.windll.winmm.timeBeginPeriod(1)
        try:
            self.stop_measurement.clear()
            self.potentiostat_disconnected.clear()
            self.set_measurement_results([[], pd.DataFrame()])
            with ps.connect() as manager:
                for segment_index, method in enumerate(methods, start=1):
                    if self.stop_measurement.is_set():
                        break
                    self.method = method
                    self.set_measurement_start_time([])
                    self.set_callback_data( {"Potential": [],
                     "Current": [],
                     "Time": []})
                    print(f"Starting CA segment {segment_index}/{len(methods)} at {method.potential} V")
                    measurement = manager.measure(method, callback=self.callback)
                    print("CA segment finished")
                    session_measurements.append(measurement)
                    measurement_results.append({
                        "current_integral": float(np.sum(np.abs(measurement.dataset.arrays(type='Current'))))
                    })
                    new_df = pd.DataFrame(measurement.dataset.to_dict())
                    if "Potential" not in new_df.columns:
                        new_df["Potential"] = method.potential
                    corrected_df = self.correct_time_stamps(self.get_measurement_start_time(), new_df)
                    corrected_df["CA_Segment"] = segment_index
                    corrected_df["CA_Segment_Potential"] = method.potential
                    results_df = self.append_measurement_frame(results_df, corrected_df)
                    self.set_measurement_results([measurement_results, results_df])
                    self.save_df_to_csv(results_df, file_name)
                manager.disconnect()
                self.potentiostat_disconnected.set()
            if session_measurements:
                self.save_measurement_to_file(session_measurements, file_name)
            print("Potentiostat disonnected")
        finally:
            self.potentiostat_disconnected.set()
            self.set_callback_data( {"Potential": [],
             "Current": [],
             "Time": []})
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
        if len(df["Potential"]) < 2:
            df["direction"] = 1
        elif self.method is not None and getattr(self.method, "id", "") == "ad":
            df["direction"] = 1
        else:
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
        csv_data = self.load_csv_file(file_name)
        if "No_Applied_Potential" in csv_data and not csv_data.empty and csv_data["No_Applied_Potential"].eq(1).all():
            run_time = float(csv_data["CA_Run_Time"].iloc[0])
            self.ca_measurement(0.0, run_time)
            self.no_applied_potential = True
            self.measurement_name = "T1 without applied potential (simulated zero current)"
            self.avg_per_point = int(csv_data["T1_Averages_Per_Datapoint"].iloc[0])
            self.n_scans = 1
            self.set_measurement_results([[{"current_integral": 0.0}], csv_data])
            self.potentiostat_disconnected.set()
            return
        self.no_applied_potential = False
        self.set_measurement_results([self.load_pssession_file(file_name),self.load_csv_file(file_name)])
        self.n_scans = len(self.get_measurement_results()[0])
        self.method=self.get_measurement_results()[0][0].method
        self.avg_per_point=1 #Dummy value
        settings = self.method.to_settings() if hasattr(self.method, "to_settings") else self.method
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
        elif self.method.id == "ad":
            self.measurement_name = "Chronoamperometry"
            self.max_points = max(int(np.ceil(settings.run_time / settings.interval_time)), 1)
            self.avg_per_point = 300
            csv_data = self.get_measurement_results()[1]
            if "CA_Segment" in csv_data.columns:
                self.measurement_name = "Combined Chronoamperometry"
                self.ca_sequence_potentials = [
                    float(value)
                    for value in csv_data.groupby("CA_Segment")["CA_Segment_Potential"].first().to_list()
                ]
                self.ca_sequence_run_time = float(settings.run_time)
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
        self.set_callback_data({"Potential": [], "Current": [], "Time": []})
        self.set_measurement_start_time([])
        self.method = None
        self.no_applied_potential = False
        self.measurement_name = ""
        self.n_scans = 0
        self.max_points = 0
        self.avg_per_point = 0
        self.t1_potential_step = 0.01
        self.ca_sequence_potentials = []
        self.ca_sequence_run_time = None
        self.set_measurement_results([[], pd.DataFrame()])
    def start_n_measurements(self,file_name,method,n_scans):
        self.stop_measurement.clear()
        self.potentiostat_disconnected.clear()
        self.set_measurement_results([[], pd.DataFrame()])
        self.set_callback_data({"Potential": [], "Current": [], "Time": []})
        self.set_measurement_start_time([])
        self.n_scans = n_scans
        thread = threading.Thread(target=self.measure_n_times, args=(file_name, method, n_scans),daemon=True)
        thread.start()
        print("EC Experiment started")
    def start_method_sequence(self,file_name,methods,avg_per_point=0):
        self.stop_measurement.clear()
        self.potentiostat_disconnected.clear()
        self.set_measurement_results([[], pd.DataFrame()])
        self.set_callback_data({"Potential": [], "Current": [], "Time": []})
        self.set_measurement_start_time([])
        self.avg_per_point = avg_per_point
        self.n_scans = len(methods)
        thread = threading.Thread(target=self.measure_method_sequence, args=(file_name, methods),daemon=True)
        thread.start()
        print("EC Experiment started")
    def start_measurement_until_stop(self,file_name,method,avg_per_point):
        self.stop_measurement.clear()
        self.potentiostat_disconnected.clear()
        self.set_measurement_results([[], pd.DataFrame()])
        self.set_callback_data({"Potential": [], "Current": [], "Time": []})
        self.set_measurement_start_time([])
        self.avg_per_point = avg_per_point
        thread = threading.Thread(target=self.measure_until_done, args=(file_name,method),daemon=True)
        thread.start()
        print("EC Experiment started")
    
  

