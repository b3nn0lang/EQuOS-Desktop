# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:29:10 2026

@author: Benno Dauer
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
from time import time
import os

def qs_data_displayer(qs_data_reader: object, fitting_parameters: dict, fig: Figure | None = None) -> tuple[object, object, bool]:
    data = qs_data_reader.evaluate_qs_data()
    
    figure_title = data["experiment"] if data["experiment"] else "QS data"

    if fig is None:
        fig = Figure(figsize=(12, 4), constrained_layout=True)
    else:
        fig.clear()
    axes = fig.subplots(1, 2)
    for ax in axes:
        ax.clear()
    fig.suptitle(figure_title)
    info_dic = dict()
    
    try:
        measurement_progress = qs_data_reader.get_raw_dataframe_progress()
        measurement_done = qs_data_reader.dataframe_loop_stop.is_set()
        x_data = np.array(data["x_data"])
        x_label = f"{data['x_label']} / {data['x_unit']}" if data["x_unit"] else data["x_label"]
        contrast_mean = np.array(data["contrast_mean"])
        contrast_std =  np.array(data["contrast_std"])
        if x_data.size == 0:
            return fig, axes, measurement_done, measurement_progress, info_dic
        info_dic["Measurement_time"] = f"{np.mean(data['time_measure_mean'])*1000:.0f} +- {np.mean(data['time_measure_std'])*1000:.0f} ms"
        info_dic["Saving time"] = f"{np.mean(data['time_save_mean'])*1000:.0f} +- {np.mean(data['time_save_std'])*1000:.0f} ms"
        info_dic["Light Level"] = f"{np.mean(data['reference_mean']):.4f} V"
        info_dic["Measurement points"] = f"{len(data['x_data'])} points x {data['contrast_length'][0]} averages"
        axes[0].plot(x_data, contrast_mean, color="tab:blue", label="contrast mean")
        """ axes[0].fill_between(
            x_data,
            [mean - std for mean, std in zip(contrast_mean, contrast_std)],
            [mean + std for mean, std in zip(contrast_mean, contrast_std)],
            color="tab:blue",
            alpha=0.2,
            label="contrast std",
        ) """#Fitting
        if fitting_parameters["fitting"] == True:
            fitting_time_begin = time()
            measurement_type = data["experiment"]
            if measurement_type == "ESR" or measurement_type == "ESR_ECQS":
                prominence = 0.01
                distance = 5
                if "esr_prominence" in fitting_parameters.keys():
                    prominence = float(fitting_parameters["esr_prominence"])
                if "esr_distance" in fitting_parameters.keys():
                    distance = float(fitting_parameters["esr_distance"])    
                peaks,_ = find_peaks(-np.array(contrast_mean), prominence=prominence, distance=distance)
                print(f"peaks found: {len(peaks)} Ghz")
                if len(peaks) > 0:
                    axes[0].plot(x_data[peaks], contrast_mean[peaks],"rx",label=f"{x_data[peaks]}")
            if measurement_type == "T1" or measurement_type == "T1_ECQS":
                if "t1_bi_exp_fit" in fitting_parameters.keys() and fitting_parameters["t1_bi_exp_fit"] == True:
                    p0 = [x_data[-1] / 10,  # tau1 - fast
                          x_data[-1] / 2,   # tau2 - slow
                          np.min(contrast_mean),  # y0 from actual first point
                          0.5]                  
                    if "t1_max_bi_tries" in fitting_parameters.keys() :
                        maxfev= fitting_parameters["t1_max_bi_tries"]
                    else:
                        maxfev= 20000
                    if "t1_use_bi_bounds" in fitting_parameters.keys() and fitting_parameters["t1_use_bi_bounds"] == True:
                       bounds = (
                           [x_data[1],       x_data[1],       0.5, 0.0],
                           [x_data[-1]*10,   x_data[-1]*10,   1.5,  1.0]
                                )
                    else:
                        bounds = (-np.inf, np.inf)
                    try:
                        min_idx = np.argmin(contrast_mean)
                        popt, pcov = curve_fit(bi_exp, x_data[min_idx:], contrast_mean[min_idx:], p0=p0, bounds=bounds, maxfev=maxfev)
                        tau = np.min([popt[0],popt[1]])
                        y_fit = bi_exp(x_data, *popt)
                        r_square = r_squared(contrast_mean[min_idx:], y_fit[min_idx:])
                        axes[0].plot([], [], label=f"tau (biexp.): {tau:.4f} ms",color="orange")
                        info_dic["Biexponential Fit"] = f"{popt[2]:.3f} + (1-{popt[2]:.3f})*( 1-({popt[3]:.3f}*e^(-x/{popt[0]:.3f}) + (1-{popt[3]:.3f})*e^(-x/{popt[1]:.3f}))) - R^2 = {r_square:.3f}"
                        if "t1_show_bi_fit" in fitting_parameters.keys() and fitting_parameters["t1_show_bi_fit"] == True:
                            axes[0].plot(x_data, y_fit, linestyle="--",color="orange", label=f"{popt[2]:.3f} + (1-{popt[2]:.3f})*( 1-({popt[3]:.3f}*e^(-x/{popt[0]:.3f}) + (1-{popt[3]:.3f})*e^(-x/{popt[1]:.3f})))")
                    except  RuntimeError as e:
                        info_dic["Bi Exp fit"] = "Fit did not work"
                        print(e)
                        
                if ("t1_stretched_exp_fit" in fitting_parameters.keys() and fitting_parameters["t1_stretched_exp_fit"]) or "t1_stretched_exp_fit" not in fitting_parameters.keys():
                    p0 = [x_data[-1] / 3,  # tau ~ 1/3 of x range
                          np.min(contrast_mean), # y0 from actual first point
                          1.0]              #beta
                                      

                    if "t1_max_stretch_tries" in fitting_parameters.keys():
                        maxfev= fitting_parameters["t1_max_stretch_tries"]
                    else:
                        maxfev= 20000
                    if "t1_use_stretch_bounds" in fitting_parameters.keys() and fitting_parameters["t1_use_stretch_bounds"] == True:
                         bounds = (
                                    [x_data[1],        0.5, 0.1],          # lower
                                    [x_data[-1] * 10,  1.5,  3.0]           # upper
                                )

                    else:
                        bounds = (-np.inf, np.inf)
                    try:
                        min_idx = np.argmin(contrast_mean)
                        popt, pcov = curve_fit(stretched_exp, x_data[min_idx:], contrast_mean[min_idx:], p0=p0, bounds=bounds, maxfev=maxfev)
                        tau = popt[0]
                        y_fit = stretched_exp(x_data, *popt)
                        r_square = r_squared(contrast_mean[min_idx:], y_fit[min_idx:])
                        axes[0].plot([], [], label=f"tau (str.exp.): {tau:.4f} ms",color="green")
                        info_dic["Stretched Exp Fit"] = f"{popt[1]:.3f} + (1-{popt[1]:.3f}) * (1-e^(-(x/{popt[0]:.3f})^{popt[2]:.3f})) - R^2 = {r_square:.3f}"
                        if "t1_show_stretch_fit" in fitting_parameters.keys() and fitting_parameters["t1_show_stretch_fit"] == True:
                            axes[0].plot(x_data, y_fit, linestyle="--",color="green", label=f"{popt[1]:.3f} + (1-{popt[1]:.3f}) * (1-e^(-(x/{popt[0]:.3f})^{popt[2]:.3f}))")
                    except  RuntimeError as e:
                         info_dic["Stretched exp fit"] = "Fit did not work"
                         print(e)
            if measurement_type == "Rabi" or measurement_type == "Rabi_ECQS":
                if ("rabi_sin_fit" in fitting_parameters.keys() and fitting_parameters["rabi_sin_fit"] == True) or "rabi_sin_fit" not in fitting_parameters.keys():
                    p0= [
                        -0.03,              # A
                        2*np.pi/200,       # B
                        -np.pi/2,          # C (inverted start)
                        np.mean(contrast_mean)# D
                        ]
                    if "rabi_max_tries" in fitting_parameters.keys():
                        maxfev = fitting_parameters["rabi_max_tries"]
                    else:
                        maxfev = 20000
                    if "rabi_use_bounds" in fitting_parameters.keys() and fitting_parameters["rabi_use_bounds"] == True:
                        bounds = (
                            [-0.2,        2*np.pi/800, -2*np.pi, 0.6],  # lower
                            [0.0,        2*np.pi/20,   0, 1.2]   # upper
                            )
                    else:
                        bounds = (-np.inf, np.inf)
                    try:
                        popt, pcov = curve_fit(
                            rabi_fit_function,
                            x_data,
                            contrast_mean,
                            p0=p0,
                            bounds=bounds,
                            maxfev=maxfev)
                        pi_pulse_duration = get_pi_from_rabi_fit(popt[1], popt[2])
                        y = np.interp(pi_pulse_duration, x_data, contrast_mean)
                        y_fit = rabi_fit_function(x_data,*popt) 
                        
                        r_square = r_squared(contrast_mean, y_fit)
                        axes[0].plot(pi_pulse_duration, y,"gx",label =f"pi pulse: {pi_pulse_duration} ns")
                        if "rabi_show_fit" in fitting_parameters.keys() and fitting_parameters["rabi_show_fit"] == True:
                            axes[0].plot(x_data,y_fit,linestyle="--",color="green", label=f"{popt[0]:.3f} *cos({popt[1]:.3f} * x + {popt[2]:.3f}) + {popt[3]:.3f}")
                        info_dic["Decaying sin fit"] = f"{popt[0]:.3f} *cos({popt[1]:.3f} * x + {popt[2]:.3f}) + {popt[3]:.3f} - R^2 = {r_square:.3f}"
                    except RuntimeError as e:
                        info_dic["Decaying sin fit"] = "Fit did not work"
                        print(e)
                if "rabi_find_peak" in fitting_parameters.keys() and fitting_parameters["rabi_find_peak"] == True:
                    prominence = 0.01
                    distance = 20
                    if "rabi_prominence" in fitting_parameters.keys():
                        prominence = float(fitting_parameters["rabi_prominence"])
                    if "rabi_distance" in fitting_parameters.keys():
                        distance = float(fitting_parameters["rabi_distance"])    
                    peaks,_ = find_peaks(-np.array(contrast_mean), prominence=prominence, distance=distance)
                    print(f"peaks found: {len(peaks)}")
                    if len(peaks) > 0:
                        axes[0].plot(x_data[peaks], contrast_mean[peaks],"rx",label=f"{x_data[peaks]} - {x_label}")
            info_dic["fitting_time"] = f"{(time()-fitting_time_begin)*1000:.0f} ms"
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel("contrast")
        axes[0].set_title("Contrast")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        
        
        reference_mean = data["reference_mean"]
        reference_std = data["reference_std"]
        value_mean = data["value_mean"]
        value_std = data["value_std"]
    
        axes[1].plot(x_data, reference_mean, color="tab:orange", label="reference mean")
        """axes[1].fill_between(
            x_data,
            [mean - std for mean, std in zip(reference_mean, reference_std)],
            [mean + std for mean, std in zip(reference_mean, reference_std)],
            color="tab:orange",
            alpha=0.2,
            label="reference std",
        )"""
        axes[1].plot(x_data, value_mean, color="tab:green", label="value mean")
        """axes[1].fill_between(
            x_data,
            [mean - std for mean, std in zip(value_mean, value_std)],
            [mean + std for mean, std in zip(value_mean, value_std)],
            color="tab:green",
            alpha=0.2,
            label="value std",
        )"""
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("light-level")
        axes[1].set_title("Reference and Value")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        #print("Image created")
    except Exception as e:
        print(f"Something in Data evaluator went wrong: {e}")
        measurement_progress = qs_data_reader.get_raw_dataframe_progress()
        measurement_done = qs_data_reader.dataframe_loop_stop.is_set()
        pass
    finally:
        return fig, axes, measurement_done, measurement_progress, info_dic
        
def rabi_fit_function(x,a,b,c,d):
    return a * np.cos(b * x + c) + d
def get_pi_from_rabi_fit(b,c):
    for i in range(-5,5,2):
        pi_fit= (np.pi - c + i*np.pi) / b
        if pi_fit>0:
            return pi_fit
def r_squared(ydata, yfit):
    ss_res = np.sum((ydata - yfit) ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    return 1 - (ss_res / ss_tot)

def stretched_exp(x,tau, y0, beta):
    return y0 + (1 - y0) * (1 - np.exp(-(x / tau)**beta))
def bi_exp(x,tau1, tau2, y0, a):
    return y0 + (1 - y0) * (
        1 - (a * np.exp(-x / tau1) + (1 - a) * np.exp(-x / tau2))
    )



def ec_data_displayer(potentiostat:object, fitting_dic:dict, fig: Figure | None = None):
    info_dic = {}
    measurement_done = False
    data = potentiostat.evaluated_data() 
    print(data)
    figure_title = potentiostat.measurement_name
    method = potentiostat.method.to_dict()
    if method["id"] == "cv":
        info_dic["Begin Potential"] = method["vertex1_potential"]
        info_dic["End Potential"] = method["vertex2_potential"]
        info_dic["Potential Steps"] = method["step_potential"]
        info_dic["Scanrate"] = method["scanrate"]
    elif method["id"] == "swv":
        info_dic["Begin Potential"] = method["begin_potential"]
        info_dic["End Potential"] = method["end_potential"]
        info_dic["Potential Steps"] = method["step_potential"]
        info_dic["Frequency"] = method["frequency"]
        info_dic["Amplitude"] = method["amplitude"]
    elif method["id"] == "lsv":
        info_dic["Begin Potential"] = method["begin_potential"]
        info_dic["End Potential"] = method["end_potential"]
        info_dic["Potential Steps"] = method["step_potential"]
        info_dic["Scanrate"] = method["scanrate"]
    #print(method)
    #info_dic["Method"] = method["General"]
    current_measurement = potentiostat.get_measurement_count()
    n_measurements = potentiostat.n_scans
    max_points = potentiostat.max_points
    info_dic["Final Length"] = f"{n_measurements} x {max_points}"
    callback_data = potentiostat.get_callback_data()
    #print(callback_data)
    info_dic["Current Length"] = f"{current_measurement} x {len(callback_data['Potential'])}"
    measurement_progress = (current_measurement*max_points + len(callback_data["Potential"])) / (n_measurements*max_points)
    if current_measurement == n_measurements:
        measurement_done = True
    if potentiostat.potentiostat_disconnected.is_set() == True:
        measurement_done=True
    if fig is None:
        fig = Figure(figsize=(12,4), constrained_layout=True)
    else:
        fig.clear()
    axes = fig.subplots(1, 2)
    fig.suptitle(figure_title)
    #Plot Callback
    try:
        axes[0].plot(callback_data['Potential'],callback_data["Current"],"gx")
    except:
        pass
    
    axes[0].set_xlabel("Potential [V]")
    axes[0].set_ylabel("Current [uA]")
    if data is not None:
        plot_up = data[data["direction"] == 1]
        axes[0].plot(plot_up["Potential"], plot_up["Current_mean"], color="tab:blue", label="Upwards Current")
        x = plot_up["Potential"].to_numpy()
        y = plot_up["Current_mean"].to_numpy()
        y_std = plot_up["Current_std"].to_numpy()
        axes[0].fill_between(
            x,
            y - y_std,
            y + y_std,
            color="tab:blue",
            alpha=0.2,
            label="Upwards Current std",
        )
        plot_down = data[data["direction"] == -1]
        axes[0].plot(plot_down["Potential"], plot_down["Current_mean"], color="tab:red", label="Downwards Current")
        x = plot_down["Potential"].to_numpy()
        y = plot_down["Current_mean"].to_numpy()
        y_std = plot_down["Current_std"].to_numpy()
        axes[0].fill_between(
            x,
            y - y_std,
            y + y_std,
            color="tab:red",
            alpha=0.2,
            label="Downwards Current std",
        )
        if fitting_dic["fitting"] == True:
            x_list = [plot_up["Potential"],plot_down["Potential"]]
            y_list = [plot_up["Current_mean"],plot_down["Current_mean"]]
            if "prominence" in fitting_dic.keys():
                prominence = fitting_dic["prominence"]
            else:
                prominence=1e-6
            if "distance" in fitting_dic.keys():
                distance = fitting_dic["distance"]
            else:
                distance = 10
            for x_s,y_s,color in zip(x_list,y_list,["b","r"]):
                x = x_s.to_numpy()
                y = y_s.to_numpy()
                # --- Positive peaks ---
                pos_peaks, pos_properties = find_peaks(
                    y,
                    prominence=prominence,   # how much peak stands out from surroundings (tune this)
                    distance=distance,       # minimum number of points between peaks (tune this)
                )
                
                # --- Negative peaks (invert the signal) ---
                neg_peaks, neg_properties = find_peaks(
                    -y,
                    prominence=prominence,
                    distance=distance,
                )
                
                # Access results
                axes[0].plot(
                x[pos_peaks], y[pos_peaks],
                f"{color}^",  # red upward triangle
                markersize=8,
                label=f"Anodic: {np.round(x[pos_peaks], 3)} V"
                )
                 
                axes[0].plot(
                x[neg_peaks], y[neg_peaks],
                f"{color}v",  # downward triangle
                markersize=8,
                label=f"Cathodic: {np.round(x[neg_peaks], 3)} V"
                )
    
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    
    integral = potentiostat.get_measurement_integrals()
    axes[1].plot(range(1, len(integral) + 1), integral,  label="Current sum")
    axes[1].set_xlabel("N Measurement")
    axes[1].set_ylabel("Abs Current Sum [A]")
    return fig, axes, measurement_done, measurement_progress, info_dic
    
def ecqs_data_displayer(qs_data_reader, potentiostat, fitting_dic: dict, cache, fig: Figure | None = None):
    if "cutoff_rate" in fitting_dic.keys():
        cutoff_rate = fitting_dic["cutoff_rate"]
    else:
        cutoff_rate = 0.5
    if "max_tries" in fitting_dic.keys():
         maxfev = fitting_dic["max_tries"]
    else:
         maxfev = 50
        
    info_dic ={}
    measurement_done = False
    kill_qupyt =False
    method = potentiostat.method.to_dict()
    qs_data = qs_data_reader.evaluate_qs_data()
    callback_data = potentiostat.get_callback_data()
   
    current_measurement = potentiostat.get_measurement_count()
    if method["id"] == "cv":
        info_dic["Begin Potential"] = method["vertex1_potential"]
        info_dic["End Potential"] = method["vertex2_potential"]
        info_dic["Potential Steps"] = method["step_potential"]
        info_dic["Scanrate"] = method["scanrate"]
        info_dic["EC: Time/Point"] = float(method["step_potential"])/float(method["scanrate"])
    elif method["id"] == "swv":
        info_dic["Begin Potential"] = method["begin_potential"]
        info_dic["End Potential"] = method["end_potential"]
        info_dic["Potential Steps"] = method["step_potential"]
        info_dic["Frequency"] = method["frequency"]
        info_dic["Amplitude"] = method["amplitude"]
        info_dic["EC: Time/Point"] = round(1/float(method["frequency"]),3)
    elif method["id"] == "lsv":
        info_dic["Begin Potential"] = method["begin_potential"]
        info_dic["End Potential"] = method["end_potential"]
        info_dic["Potential Steps"] = method["step_potential"]
        info_dic["Scanrate"] = method["scanrate"]
        
    info_dic["EC:Current Length"] = f"{current_measurement} x {len(callback_data['Potential'])}"
    try:
        info_dic["QS:Measurement_time"] = f"{np.mean(qs_data['time_measure_mean'])*1000:.0f} +- {np.mean(qs_data['time_measure_std'])*1000:.0f} ms"
        info_dic["QS:Saving time"] = f"{np.mean(qs_data['time_save_mean'])*1000:.0f} +- {np.mean(qs_data['time_save_std'])*1000:.0f} ms"
        info_dic["QS:Light Level"] = f"{np.mean(qs_data['reference_mean']):.4f} V"
        info_dic["QS:Measurement points"] = f"{len(qs_data['x_data'])} points x {qs_data['contrast_length'][0]} averages"
    except:
        pass
    
    sub_path = os.path.join(qs_data_reader.file_path, qs_data_reader.file_name)
    os.makedirs(sub_path,exist_ok=True)
    all_up_fig = Figure(figsize=(12,8), constrained_layout=True)
    all_up_ax = all_up_fig.subplots()
    all_down_fig = Figure(figsize=(12,8), constrained_layout=True)
    all_down_ax = all_down_fig.subplots()
    
    ec_data = potentiostat.evaluated_data()
    #print(f"EC data: {ec_data}")
    ec_figure_title = potentiostat.measurement_name
    qs_figure_title = qs_data["experiment"] if qs_data["experiment"] else "QS data"
    
    if fig is None:
        fig = Figure(figsize=(12,8), constrained_layout=True)
        #axes = fig.subplots(2, 2)

    fig.suptitle(f"{ec_figure_title} / {qs_figure_title}")
    
   
    
    if ec_data is not None and current_measurement > cache.last_measurement: 
        fig.clear()
        cache.last_measurement = current_measurement
        axes = fig.subplots(2, 2)
        ax_light_with_pot = axes[1,1].twinx()
        ax_light_with_pot.spines["right"].set_position(("outward", 60))
        ax_light_with_pot.set_ylabel("Lightlevel [mV]")
        time_loop_begin=time()
        try:
            plot_up = ec_data[ec_data["direction"] == 1]
            points_up = []
            if not plot_up.empty:
                axes[0,0].plot(plot_up["Potential"], plot_up["Current_mean"], color="tab:blue", label="Upwards Current")
                x = plot_up["Potential"].to_numpy()
                y = plot_up["Current_mean"].to_numpy()
                y_std = plot_up["Current_std"].to_numpy()
                axes[0,0].fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    color="tab:blue",
                    alpha=0.2,
                    label="Upwards Current std",
                )
            
                tau_values_up = []
                r_square_up = []
                x_values_up = plot_up["Potential"]
                qs_light_up = []
                print(f"v_values_up length: {len(x_values_up)}")
                all_up_ax.set_prop_cycle(color=mpl.colormaps['Blues'](np.linspace(0.2, 1.0, len(plot_up))))
                for i,row in plot_up.iterrows():
                    time_begin = np.atleast_1d(row["Time_begin"])
                    time_end   = np.atleast_1d(row["Time_end"])
                    time_frames_up = np.column_stack((time_begin,time_end))
                   # print("shape:", time_frames_up.shape)
                    #print(time_frames_up)
                    qs_data_up = qs_data_reader.evaluate_qs_data(cutoff_rate=cutoff_rate,time_array=time_frames_up)
                    qs_tau = qs_data_up["x_data"]
                    qs_contrast = qs_data_up["contrast_mean"]
                    qs_lightlevel = np.mean(qs_data_up["reference_mean"])*10
                    
                    try:
                        qs_tau = np.atleast_1d(np.asarray(qs_tau, dtype=float))
                        qs_contrast = np.atleast_1d(np.asarray(qs_contrast, dtype=float))
                    except (TypeError, ValueError):
                        qs_tau = np.array([])
                        qs_contrast = np.array([])
                    if len(qs_contrast) <3 or len(qs_tau) <3:
                        tau_values_up.append(0)
                        r_square_up.append(0)
                        points_up.append(0)
                        continue
                    #print("Contrast length:", len(qs_contrast))
                   # print(len(qs_tau))
                    cl = qs_data_up["contrast_length"]
                    if isinstance(cl, (list, np.ndarray)) and len(cl) > 0:
                        qs_points = cl[0]
                    elif isinstance(cl, (int, float)):
                        qs_points = cl
                    else:
                        qs_points = 0
                    
                    p0 = [qs_tau[-1] / 3,  # tau ~ 1/3 of x range
                          np.min(qs_contrast), # y0 from actual first point
                          1.0              # beta
                          ]                #Amp
                    if "use_bounds" in fitting_dic.keys() and fitting_dic["use_bounds"]==True:
                        bounds = (
                            [qs_tau[1],        0.5, 0.1],          # lower
                            [qs_tau[-1] * 10,  1.5,  3.0]           # upper
                        )
                    else:
                        bounds = (-np.inf, np.inf)
                    #time_0=time()
                    try:
                        min_idx = np.argmin(qs_contrast)
                        popt, pcov = curve_fit(stretched_exp, qs_tau[min_idx:], qs_contrast[min_idx:], p0=p0, bounds=bounds,maxfev=maxfev)
                        tau = popt[0]
                        if tau < 0:
                            tau =0
                        tau_values_up.append(tau)
                        y_fit = stretched_exp(qs_tau[min_idx:], *popt)
                        r_square = r_squared(qs_contrast[min_idx:], y_fit)
                        r_square_up.append(r_square)
                        """
                        Plot figure for every potential
                        """
                        plt.plot(qs_tau,qs_contrast,marker="x",color="blue",label=f"{x_values_up[i]:.3f}V")
                        all_up_ax.plot(qs_tau,qs_contrast,marker="x",label=f"{x_values_up[i]:.3f}V")
                        plt.plot(qs_tau[min_idx:],y_fit,linestyle="--",color="blue",label=f"fit: {tau:.3f} ms, R^2 = {r_square:.3f}")
                        name = f"QS_Up_{x_values_up[i]:.3f}V.png"
                        fig_path = os.path.join(sub_path,name)
                        plt.legend()
                        plt.savefig(fig_path, dpi=300)
                        plt.close()
                    except Exception as e:
                        print(e)
                        tau_values_up.append(0)
                        r_square_up.append(0)
                    finally:
                        points_up.append(qs_points)
                        qs_light_up.append(qs_lightlevel)
                  #  print(f"Up-fittingtime:{(time()-time_0)*1000} ms")
                try:
                    axes[0,1].plot(x_values_up,tau_values_up,color="blue",label="T1 Time Upwards current")
                except Exception as e:
                    print(e)
                try:
                    axes[1,1].plot(x_values_up,r_square_up,color="blue",linestyle="--",label="R^2 T1")
                except Exception as e:
                    print(e)
                try:
                    ax_light_with_pot.plot(x_values_up, qs_light_up, color="blue", label = "Lightlevel")
                except Exception as e:
                    print(e)
        except Exception as e:
            print(f"Something in plot_up failed {e}")
        
        try:
            plot_down = ec_data[ec_data["direction"] == -1]
            points_down = []
            if not plot_down.empty:
                axes[0,0].plot(plot_down["Potential"], plot_down["Current_mean"], color="tab:red", label="Downwards Current")
                x = plot_down["Potential"].to_numpy()
                y = plot_down["Current_mean"].to_numpy()
                y_std = plot_down["Current_std"].to_numpy()
                axes[0,0].fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    color="tab:red",
                    alpha=0.2,
                    label="Downwards Current std",
                )
                tau_values_down = []
                r_square_down = []
                x_values_down = plot_down["Potential"]
                qs_light_down = []
                print(f"v_values_down length: {len(x_values_down)}")
                all_down_ax.set_prop_cycle(color=mpl.colormaps['Reds'](np.linspace(0.2, 1.0, len(plot_down))))
                for i,row in plot_down.iterrows():
                    time_begin = np.atleast_1d(row["Time_begin"])
                    time_end   = np.atleast_1d(row["Time_end"])
                    time_frames_down = np.column_stack((time_begin,time_end))
                   # print("shape:", time_frames_down.shape)
                    qs_data_down = qs_data_reader.evaluate_qs_data(cutoff_rate=cutoff_rate,time_array=time_frames_down)
                    qs_tau = qs_data_down["x_data"]
                    qs_contrast = qs_data_down["contrast_mean"]
                    qs_lightlevel = np.mean(qs_data_down["reference_mean"])*10
                    try:
                        qs_tau = np.atleast_1d(np.asarray(qs_tau, dtype=float))
                        qs_contrast = np.atleast_1d(np.asarray(qs_contrast, dtype=float))
                    except (TypeError, ValueError):
                        qs_tau = np.array([])
                        qs_contrast = np.array([])
                    if len(qs_contrast) == 0 or len(qs_tau) == 0:
                        tau_values_down.append(0)
                        r_square_down.append(0)
                        points_down.append(0)
                        continue
                   # print("Contrast length: ", len(qs_contrast))
                   # print(len(qs_tau))
                    cl = qs_data_down["contrast_length"]
                    if isinstance(cl, (list, np.ndarray)) and len(cl) > 0:
                        qs_points = cl[0]
                    elif isinstance(cl, (int, float)):
                        qs_points = cl
                    else:
                        qs_points = 0
                       
                    
                    p0 = [qs_tau[-1] / 3,  # tau ~ 1/3 of x range
                          np.min(qs_contrast), # y0 from actual first point
                          1.0                    # beta
                          ]                      
                    if "use_bounds" in fitting_dic.keys() and fitting_dic["use_bounds"]==True:
                        bounds = (
                            [qs_tau[1],        0.5, 0.1],          # lower
                            [qs_tau[-1] * 10,  1.5,  3.0]           # upper
                        )
                    else:
                        bounds = (-np.inf, np.inf)
                   # t0 = time()
                    try:
                        min_idx = np.argmin(qs_contrast)
                        popt, pcov = curve_fit(stretched_exp, qs_tau[min_idx:], qs_contrast[min_idx:], p0=p0,bounds=bounds, maxfev=maxfev)
                        tau = popt[0]
                        if tau < 0:
                            tau =0
                        tau_values_down.append(tau)
                        y_fit = stretched_exp(qs_tau[min_idx:], *popt)
                        r_square = r_squared(qs_contrast[min_idx:], y_fit)
                        r_square_down.append(r_square)
                        """
                        Plot figure for every potential
                        """
                        plt.plot(qs_tau,qs_contrast,color="red",marker="x",label=f"{x_values_down[i]:.3f}")
                        all_down_ax.plot(qs_tau,qs_contrast,marker="x",label=f"{x_values_down[i]:.3f}")
                        plt.plot(qs_tau[min_idx:],y_fit,linestyle="--",color="red",label=f"fit: {tau:.3f} ms, R^2 = {r_square:.3f}")
                        name = f"QS_Down_{x_values_down[i]:.3f}V.png"
                        fig_path = os.path.join(sub_path,name)
                        plt.legend()
                        plt.savefig(fig_path, dpi=300)
                        plt.close()
                        
                    except Exception as e:
                        print(e)
                        tau_values_down.append(0)
                        r_square_down.append(0)
                    finally:
                        points_down.append(qs_points)
                        qs_light_down.append(qs_lightlevel)
                #print(f"fitting_time: {(time()-t0)*1000} ms")
                try:
                    axes[0,1].plot(x_values_down,tau_values_down,color="red",label="T1 Time Downwards current")
                except Exception as e:
                    print(e)
                try:
                    axes[1,1].plot(x_values_down,r_square_down,color="red",linestyle="--",label="R^2 T1")
                except Exception as e:
                    print(e)
                try:
                    
                    ax_light_with_pot.plot(x_values_down, qs_light_down, color="red", label = "Lightlevel")
                except Exception as e:
                    print(e)
        except Exception as e:
            print(f"Something in plot_down failed {e}")
        axes[1,1].set_ylim(0, 1.1)  # R² is always 0-1
        axes[1,1].set_ylabel("R² / [V:100]")
        axes[1,1].set_xlabel("Potential [V]")
        
        # For the twin axis, set its own scale separately:
        ax_bar = axes[1,1].twinx()
        ax_bar.set_ylabel("Averages")
        try:
            ax_bar.bar(x_values_up - float(method["step_potential"])/4, points_up, float(method["step_potential"])/2, label="Upwards Averages",color="blue",alpha=0.6)
        finally:
            try:
                ax_bar.bar(x_values_down + float(method["step_potential"])/4, points_down, float(method["step_potential"])/2, label="Downwards Averages",color="red",alpha=0.6)
            except:
                pass
        # Let bar axis autoscale independently, don't let it affect R² axis
        ax_bar.set_zorder(axes[1,1].get_zorder() - 1)  # put bars behind the R² line
        axes[1,1].patch.set_visible(False)  # make axes[1,1] background transparent so bars show through    
        print(f"Complete loop time: {time()-time_loop_begin}")
        if "ec_fitting" in fitting_dic.keys() and fitting_dic["ec_fitting"] == True:
            x_list = [plot_up["Potential"],plot_down["Potential"]]
            y_list = [plot_up["Current_mean"],plot_down["Current_mean"]]
            if "prominence" in fitting_dic.keys():
                prominence = fitting_dic["prominence"]
            else:
                prominence=1e-6
            if "distance" in fitting_dic.keys():
                distance = fitting_dic["distance"]
            else:
                distance = 10
            for x_s,y_s,color in zip(x_list,y_list,["b","r"]):
                x = x_s.to_numpy()
                y = y_s.to_numpy()
                # --- Positive peaks ---
                pos_peaks, pos_properties = find_peaks(
                    y,
                    prominence=prominence,   # how much peak stands out from surroundings (tune this)
                    distance=distance,       # minimum number of points between peaks (tune this)
                )
                
                # --- Negative peaks (invert the signal) ---
                neg_peaks, neg_properties = find_peaks(
                    -y,
                    prominence=prominence,
                    distance=distance,
                )
                
                # Access results
                axes[0,0].plot(
                x[pos_peaks], y[pos_peaks],
                f"{color}^",  # red upward triangle
                markersize=8,
                label=f"Anodic: {np.round(x[pos_peaks], 3)} V"
                )
                 
                axes[0,0].plot(
                x[neg_peaks], y[neg_peaks],
                f"{color}v",  # downward triangle
                markersize=8,
                label=f"Cathodic: {np.round(x[neg_peaks], 3)} V")
                
        integral = potentiostat.get_measurement_integrals()
        axes[1,0].plot(range(1, len(integral) + 1), integral,  label="Current sum")    
        
        #Get all EC figures
        for i,figure in enumerate(potentiostat.get_measurement_plots()):
            fig_path = os.path.join(sub_path,f"EC_{i}.png")
            if os.path.exists(fig_path):
                pass
            else:
                figure.savefig(fig_path, dpi=300)
            plt.close(figure)
            
        up_fig_name = "QS_Up_all.png"
        down_fig_name = "QS_Down_all.png"
        
        all_up_ax.legend(loc="se")
        all_down_ax.legend(loc="se")
        all_up_fig.savefig(os.path.join(sub_path, up_fig_name), dpi=300)
        all_down_fig.savefig(os.path.join(sub_path, down_fig_name), dpi=300)
        plt.close(all_up_fig)
        plt.close(all_down_fig)
        #Code lightlevel over time here
        raw_ec_data = potentiostat.get_measurement_results()[1]
        time_min = np.min(raw_ec_data["Time_begin"])
        time_max = np.max(raw_ec_data["Time_end"])
        ll_list = []
        time_lists = np.linspace(time_min, time_max, current_measurement*2)
        x_list = np.linspace(1, current_measurement,current_measurement*2)
        ax_ll = axes[1,0].twinx()
        ax_ll.set_ylabel("Lightlevel [mV]")
        for i in range(len(time_lists)-1):
            time_array=np.array([[time_lists[i], time_lists[i+1]]])
            qs_data_i = qs_data_reader.evaluate_qs_data(time_array=time_array)
            ll_i = np.mean(qs_data_i["reference_mean"])
            ll_list.append(ll_i)             
        ax_ll.plot(x_list[1:],ll_list, label="Lightlevel",color="green")
        del ll_list,x_list
        
        if not points_up and not points_down:
            measurement_progress=0
        elif points_up and not points_down:
            measurement_progress = safe_min(points_up)/potentiostat.avg_per_point
        elif not points_up and points_down:
            measurement_progress = safe_min(points_down)/potentiostat.avg_per_point
        elif points_up and points_down:
            measurement_progress = np.min([safe_min(points_down),safe_min(points_up)])/potentiostat.avg_per_point
        if measurement_progress >=1:
            #potentiostat.stop_measurement.set()
            kill_qupyt=True
        cache.axes = axes
        cache.measurement_progress = measurement_progress
        cache.kill_qupyt = kill_qupyt
        cache.fig = fig
    elif ec_data is not None:
        axes = cache.axes
        fig = cache.fig
        measurement_progress = cache.measurement_progress
        kill_qupyt = cache.kill_qupyt
    else:
        fig.clear()
        measurement_progress=0
        axes = fig.subplots(2, 2)
    #Plot Callback
    try:
        axes[0,0].plot(callback_data['Potential'],callback_data["Current"],"gx")
    except:
        pass
    axes[0,0].set_xlabel("Potential [V]")
    axes[0,0].set_ylabel("Current [uA]")
    axes[0,1].set_xlabel("Potential [V]")
    axes[0,1].set_ylabel("T1 Time")
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    axes[1,0].set_xlabel("N Measurement")
    axes[1,0].set_ylabel("Abs Current Sum [uA]")
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    if potentiostat.potentiostat_disconnected.is_set() and qs_data_reader.dataframe_loop_stop.is_set():
        measurement_done=True
    elif qs_data_reader.dataframe_loop_stop.is_set() and not potentiostat.stop_measurement.is_set():
        potentiostat.stop_measurement.set()

    return fig, axes, measurement_done, measurement_progress, info_dic,kill_qupyt

def safe_min(arr):
    arr = arr[1:-1]
    if len(arr) == 0:
        return 0
    return np.min(arr)
