# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:29:10 2026

@author: Benno Dauer
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, OptimizeWarning
import numpy as np
from time import time
import os
import glob
import warnings
from tqdm import tqdm
from Plot_export import export_figure_data_csv

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
        reference_mean = np.array(data["reference_mean"])
        reference_std = np.array(data["reference_std"])
        value_mean = np.array(data["value_mean"])
        value_std = np.array(data["value_std"])
        if data["experiment"] == "T1" or data["experiment"] == "T1_ECQS":
            contrast_mean, contrast_std, t1_cutoff_idx = normalize_averaged_decay(
                data["contrast_mean"],
                data["contrast_std"],
                return_cutoff=True,
            )
            x_data = x_data[t1_cutoff_idx:]
            reference_mean = reference_mean[t1_cutoff_idx:]
            reference_std = reference_std[t1_cutoff_idx:]
            value_mean = value_mean[t1_cutoff_idx:]
            value_std = value_std[t1_cutoff_idx:]
            info_dic["T1 normalization cutoff"] = f"{t1_cutoff_idx} points"
        else:
            contrast_mean = np.array(data["contrast_mean"])
            contrast_std =  np.array(data["contrast_std"])
        if x_data.size == 0:
            return fig, axes, measurement_done, measurement_progress, info_dic
        info_dic["Measurement_time"] = f"{np.mean(data['time_measure_mean'])*1000:.0f} +- {np.mean(data['time_measure_std'])*1000:.0f} ms"
        info_dic["Saving time"] = f"{np.mean(data['time_save_mean'])*1000:.0f} +- {np.mean(data['time_save_std'])*1000:.0f} ms"
        info_dic["Light Level"] = f"{np.mean(data['reference_mean']):.4f} mV"
        info_dic["Measurement points"] = f"{len(data['x_data'])} points x {data['contrast_length'][0]} averages"
        add_qs_measurement_parameters(info_dic, qs_data_reader)
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
                    min_tau = positive_tau_lower_bound(x_data)
                    p0 = [x_data[-1] / 10,  # tau1 - fast
                          x_data[-1] / 2,   # tau2 - slow
                          0.5]              # fast-component weight
                    if "t1_max_bi_tries" in fitting_parameters.keys() :
                        maxfev= fitting_parameters["t1_max_bi_tries"]
                    else:
                        maxfev= 20000
                    if "t1_use_bi_bounds" in fitting_parameters.keys() and fitting_parameters["t1_use_bi_bounds"] == True:
                       bounds = (
                           [min_tau,         min_tau,         0.0],
                           [x_data[-1]*10,   x_data[-1]*10,   1.0]
                                )
                    else:
                        bounds = (-np.inf, np.inf)
                    try:
                        popt, pcov = curve_fit(bi_exp, x_data, contrast_mean, p0=p0, bounds=bounds, maxfev=maxfev)
                        tau = np.min([popt[0],popt[1]])
                        y_fit = bi_exp(x_data, *popt)
                        r_square = r_squared(contrast_mean, y_fit)
                        axes[0].plot([], [], label=f"tau (biexp.): {tau:.4f} ms",color="orange")
                        info_dic["Biexponential Fit"] = f"{popt[2]:.3f}*e^(-x/{popt[0]:.3f}) + (1-{popt[2]:.3f})*e^(-x/{popt[1]:.3f}) - R^2 = {r_square:.3f}"
                        if "t1_show_bi_fit" in fitting_parameters.keys() and fitting_parameters["t1_show_bi_fit"] == True:
                            axes[0].plot(x_data, y_fit, linestyle="--",color="orange", label=f"{popt[2]:.3f}*e^(-x/{popt[0]:.3f}) + (1-{popt[2]:.3f})*e^(-x/{popt[1]:.3f})")
                    except  RuntimeError as e:
                        info_dic["Bi Exp fit"] = "Fit did not work"
                        print(e)
                        
                if ("t1_stretched_exp_fit" in fitting_parameters.keys() and fitting_parameters["t1_stretched_exp_fit"]) or "t1_stretched_exp_fit" not in fitting_parameters.keys():
                    min_tau = positive_tau_lower_bound(x_data)
                    p0 = [x_data[-1] / 3,  # tau ~ 1/3 of x range
                          1.0]             # beta
                                      

                    if "t1_max_stretch_tries" in fitting_parameters.keys():
                        maxfev= fitting_parameters["t1_max_stretch_tries"]
                    else:
                        maxfev= 20000
                    if "t1_use_stretch_bounds" in fitting_parameters.keys() and fitting_parameters["t1_use_stretch_bounds"] == True:
                         bounds = (
                                    [min_tau,          0.1],          # lower
                                    [x_data[-1] * 10,  3.0]           # upper
                                )

                    else:
                        bounds = (-np.inf, np.inf)
                    try:
                        popt, pcov = curve_fit(stretched_exp, x_data, contrast_mean, p0=p0, bounds=bounds, maxfev=maxfev)
                        tau = popt[0]
                        y_fit = stretched_exp(x_data, *popt)
                        r_square = r_squared(contrast_mean, y_fit)
                        axes[0].plot([], [], label=f"tau (str.exp.): {tau:.4f} ms",color="green")
                        info_dic["Stretched Exp Fit"] = f"e^(-(x/{popt[0]:.3f})^{popt[1]:.3f}) - R^2 = {r_square:.3f}"
                        if "t1_show_stretch_fit" in fitting_parameters.keys() and fitting_parameters["t1_show_stretch_fit"] == True:
                            axes[0].plot(x_data, y_fit, linestyle="--",color="green", label=f"e^(-(x/{popt[0]:.3f})^{popt[1]:.3f})")
                    except  RuntimeError as e:
                         info_dic["Stretched exp fit"] = "Fit did not work"
                         print(e)
            if measurement_type == "Rabi" or measurement_type == "Rabi_ECQS":
                if ("rabi_sin_fit" in fitting_parameters.keys() and fitting_parameters["rabi_sin_fit"] == True) or "rabi_sin_fit" not in fitting_parameters.keys():
                    rabi_fit_skip_points = 5
                    x_fit_data = x_data[rabi_fit_skip_points:]
                    contrast_fit_mean = contrast_mean[rabi_fit_skip_points:]
                    info_dic["Rabi fit skipped"] = "first 10 ns / 5 points"
                    p0 = estimate_rabi_initial_guess(x_fit_data, contrast_fit_mean)
                    if "rabi_max_tries" in fitting_parameters.keys():
                        maxfev = fitting_parameters["rabi_max_tries"]
                    else:
                        maxfev = 20000
                    if "rabi_use_bounds" in fitting_parameters.keys() and fitting_parameters["rabi_use_bounds"] == True:
                        bounds = rabi_fit_bounds(x_fit_data, contrast_fit_mean)
                    else:
                        bounds = (-np.inf, np.inf)
                    try:
                      #  weights = np.exp(-x_data / (x_data[-1] / 1))   # decay scale ~ 1/3 of x range
                       # sigma = 1.0 / (weights + 1e-10)                 # invert: high weight → small sigma
                        popt, pcov = curve_fit(
                            rabi_fit_function,
                            x_fit_data,
                            contrast_fit_mean,
                            p0=p0,
                            bounds=bounds,
                            maxfev=maxfev
                            #sigma=sigma,
                           # absolute_sigma=False
                           )
                        #pi_pulse_duration = get_pi_from_rabi_fit(popt[1], popt[2]) OLD
                        y_fit = rabi_fit_function(x_data,*popt) 
                        y_fit_for_score = rabi_fit_function(x_fit_data,*popt)
                        pi_pulse_duration = first_rabi_dip_from_fit(x_fit_data, popt)
                        y = np.interp(pi_pulse_duration, x_data, contrast_mean)
                        
                        r_square = r_squared(contrast_fit_mean, y_fit_for_score)
                        axes[0].plot(pi_pulse_duration, y,"gx",label =f"pi pulse: {pi_pulse_duration} ns")
                        if "rabi_show_fit" in fitting_parameters.keys() and fitting_parameters["rabi_show_fit"] == True:
                            axes[0].plot(x_data,y_fit,linestyle="--",color="green", label=f"{popt[0]:.3f} * exp(-x/{popt[3]:.0f}) * cos({popt[1]:.3f} * x + {popt[2]:.3f}) + {popt[4]:.3f}")
                        info_dic["Decaying sin fit"] = f"{popt[0]:.3f} * exp(-x/{popt[3]:.0f}) * cos({popt[1]:.3f} * x + {popt[2]:.3f}) + {popt[4]:.3f} - R^2 = {r_square:.3f}"
                        info_dic["Pi pulse duration"] = f"{pi_pulse_duration:.2f} ns"
                    except (RuntimeError, ValueError) as e:
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
        axes[1].set_ylabel("light-level [mV]")
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
"""        
def rabi_fit_function(x,a,b,c,d):
    return a * np.cos(b * x + c) + d
"""
def rabi_fit_function(x, A, omega, phi, T, offset):
    return A * np.exp(-x/T) * np.cos(omega*x + phi) + offset

def estimate_rabi_initial_guess(x_data, contrast_mean):
    x_data = np.asarray(x_data, dtype=float)
    contrast_mean = np.asarray(contrast_mean, dtype=float)
    x_span = max(float(np.ptp(x_data)), 1.0)
    y_span = max(float(np.ptp(contrast_mean)), 0.002)
    offset = float(np.median(contrast_mean))
    amplitude = min(max(y_span / 2, 0.005), 0.12)
    dips, _ = find_peaks(-contrast_mean)
    if len(dips) > 0:
        first_dip = max(float(x_data[dips[0]]), 1.0)
        period = min(max(first_dip * 2, 35.0), 300.0)
    else:
        period = 50.0
    return [
        amplitude,
        2 * np.pi / period,
        0.0,
        max(x_span, 250.0),
        offset,
    ]

def rabi_fit_bounds(x_data, contrast_mean):
    x_data = np.asarray(x_data, dtype=float)
    contrast_mean = np.asarray(contrast_mean, dtype=float)
    x_span = max(float(np.ptp(x_data)), 1.0)
    y_min = float(np.min(contrast_mean))
    y_max = float(np.max(contrast_mean))
    y_span = max(y_max - y_min, 0.01)
    return (
        [
            0.0005,          # A
            2 * np.pi / 300, # period up to 300 ns
            -2 * np.pi,      # phi
            10,              # T
            y_min - y_span,  # offset
        ],
        [
            0.20,            # A, enough for about 8-10% contrast and beyond
            2 * np.pi / 10,  # period down to 10 ns
            2 * np.pi,       # phi
            max(5000, x_span * 20),
            y_max + y_span,  # offset
        ],
    )

def first_rabi_dip_from_fit(x_data, popt):
    x_data = np.asarray(x_data, dtype=float)
    dense_points = max(1000, len(x_data) * 20)
    x_fit_dense = np.linspace(float(np.min(x_data)), float(np.max(x_data)), dense_points)
    y_fit_dense = rabi_fit_function(x_fit_dense, *popt)
    dips, _ = find_peaks(-y_fit_dense)
    dips = [dip for dip in dips if x_fit_dense[dip] > x_fit_dense[0]]
    if dips:
        return float(x_fit_dense[dips[0]])
    return float(x_fit_dense[int(np.argmin(y_fit_dense))])

"""def get_pi_from_rabi_fit(b,c):
    for i in range(-5,5,2):
        pi_fit= (np.pi - c + i*np.pi) / b
        if pi_fit>0:
            return pi_fit
"""
def r_squared(ydata, yfit):
    ss_res = np.sum((ydata - yfit) ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    if ss_tot == 0:
        return 0
    return 1 - (ss_res / ss_tot)

def axes_at_same_position(ax):
    base_bounds = ax.get_position().bounds
    return [
        candidate
        for candidate in ax.figure.axes
        if np.allclose(candidate.get_position().bounds, base_bounds)
    ]

def combined_legend(ax, loc="best"):
    handles = []
    labels = []
    for candidate in axes_at_same_position(ax):
        candidate_handles, candidate_labels = candidate.get_legend_handles_labels()
        for handle, label in zip(candidate_handles, candidate_labels):
            if not label or label.startswith("_") or label in labels:
                continue
            handles.append(handle)
            labels.append(label)
    if handles:
        ax.legend(handles, labels, loc=loc)

def plot_line_with_missing_fit_markers(ax, x_values, y_values, color, label):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(y_values)
    missing = ~valid

    if np.any(valid):
        ax.plot(x_values[valid], y_values[valid], marker="x",color=color, label=label)
    if np.any(missing) and np.any(valid):
        sort_idx = np.argsort(x_values[valid])
        marker_y = np.interp(
            x_values[missing],
            x_values[valid][sort_idx],
            y_values[valid][sort_idx],
        )
        ax.scatter(
            x_values[missing],
            marker_y,
            marker="o",
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
            label=f"{label} missing fit",
            zorder=4,
        )

def set_t1_axis_from_zero(ax):
    y_values = []
    for line in ax.lines:
        y_data = np.asarray(line.get_ydata(), dtype=float)
        y_values.extend(y_data[np.isfinite(y_data)])

    if not y_values:
        ax.set_ylim(bottom=0)
        return

    y_max = max(y_values)
    if y_max <= 0:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, y_max * 1.05)

def save_ecqs_t1_detail_plot(qs_tau, qs_contrast, potential, color, direction, sub_path, all_ax, y_fit=None, tau=None, r_square=None, fit_error=None):
    qs_tau = np.asarray(qs_tau, dtype=float)
    qs_contrast = np.asarray(qs_contrast, dtype=float)
    label = f"{potential:.3f}V"
    name = f"QS_{direction}_{potential:.3f}V.png"
    fig_path = os.path.join(sub_path, name)
    fig = Figure(figsize=(6, 4), constrained_layout=True)
    ax = fig.subplots()

    if len(qs_tau) and len(qs_contrast):
        ax.plot(qs_tau, qs_contrast, marker="x", color=color, label=label)
        all_ax.plot(qs_tau, qs_contrast, marker="x", label=label)
    else:
        ax.plot([], [], marker="x", color=color, label=f"{label}: no T1 data")

    if y_fit is not None and tau is not None and r_square is not None:
        ax.plot(
            qs_tau,
            y_fit,
            linestyle="--",
            color=color,
            label=f"fit: {tau:.3f} ms, R^2 = {r_square:.3f}",
        )
    elif fit_error:
        ax.plot([], [], color=color, label=f"fit failed: {fit_error}")

    ax.set_xlabel("tau spacing")
    ax.set_ylabel("normalized contrast")
    ax.legend()
    fig.savefig(fig_path, dpi=300)
    export_figure_data_csv(fig, os.path.splitext(fig_path)[0] + ".csv")
    plt.close(fig)

def clear_ca_t1_detail_plots(sub_path):
    old_files = glob.glob(os.path.join(sub_path, "CA_T1_point_*.png"))
    old_files += glob.glob(os.path.join(sub_path, "CA_T1_point_*.csv"))
    for old_file in old_files:
        try:
            os.remove(old_file)
        except OSError:
            pass

def save_ca_t1_detail_plot(point_number, qs_tau, qs_contrast, sub_path, y_fit=None, tau=None, r_square=None, fit_error=None):
    qs_tau = np.asarray(qs_tau, dtype=float)
    qs_contrast = np.asarray(qs_contrast, dtype=float)
    n_raw = min(len(qs_tau), len(qs_contrast))
    qs_tau = qs_tau[:n_raw]
    qs_contrast = qs_contrast[:n_raw]
    fig_path = os.path.join(sub_path, f"CA_T1_point_{point_number:03d}.png")
    label = f"Point {point_number}"
    fig = Figure(figsize=(6, 4), constrained_layout=True)
    ax = fig.subplots()

    if n_raw:
        ax.plot(qs_tau, qs_contrast, marker="x", color="blue", label=label)
    else:
        ax.plot([], [], marker="x", color="blue", label=f"{label}: no T1 data")

    if y_fit is not None and tau is not None and r_square is not None and len(y_fit) == n_raw:
        ax.plot(
            qs_tau,
            y_fit,
            linestyle="--",
            color="blue",
            label=f"fit: {tau:.3f} ms, R^2 = {r_square:.3f}",
        )
    elif fit_error:
        ax.plot([], [], color="blue", label=f"fit failed: {fit_error}")

    ax.set_xlabel("tau spacing")
    ax.set_ylabel("normalized contrast")
    ax.legend()
    fig.savefig(fig_path, dpi=300)
    export_figure_data_csv(fig, os.path.splitext(fig_path)[0] + ".csv")
    plt.close(fig)

def ecqs_t1_selected_points(plot_data, t1_potential_step: float) -> list[dict]:
    if plot_data is None or plot_data.empty:
        return []
    try:
        t1_potential_step = abs(float(t1_potential_step))
    except (TypeError, ValueError):
        t1_potential_step = 0.05
    if not np.isfinite(t1_potential_step) or t1_potential_step <= 0:
        t1_potential_step = 0.05

    ordered = plot_data.copy()
    ordered["_time_mid"] = ordered.apply(
        lambda row: float(
            np.mean(
                np.concatenate((
                    np.atleast_1d(row["Time_begin"]).astype(float),
                    np.atleast_1d(row["Time_end"]).astype(float),
                ))
            )
        ),
        axis=1,
    )
    ordered = ordered.sort_values("_time_mid").reset_index(drop=True)
    potentials = ordered["Potential"].to_numpy(dtype=float)
    if len(potentials) == 0:
        return []

    sweep_sign = 1 if potentials[-1] >= potentials[0] else -1
    scan_step = max(float(np.nanmedian(np.abs(np.diff(potentials)))) if len(potentials) > 1 else 0.01, 1e-9)
    target_count = int(np.floor(abs(potentials[-1] - potentials[0]) / t1_potential_step)) + 1
    targets = potentials[0] + sweep_sign * t1_potential_step * np.arange(target_count + 1)
    end_guard = potentials[-1] + sweep_sign * scan_step * 0.5
    targets = targets[targets <= end_guard] if sweep_sign > 0 else targets[targets >= end_guard]
    if len(targets) == 0:
        targets = np.array([potentials[0]])

    selected_indices = []
    for target in targets:
        nearest = int(np.argmin(np.abs(potentials - target)))
        if nearest not in selected_indices:
            selected_indices.append(nearest)
    if selected_indices[-1] != len(ordered) - 1 and abs(potentials[-1] - potentials[selected_indices[-1]]) >= t1_potential_step * 0.5:
        selected_indices.append(len(ordered) - 1)

    selected = []
    for pos, idx in enumerate(selected_indices):
        prev_idx = selected_indices[pos - 1] if pos > 0 else None
        next_idx = selected_indices[pos + 1] if pos + 1 < len(selected_indices) else None
        start_idx = 0 if prev_idx is None else int(np.floor((prev_idx + idx) / 2)) + 1
        end_idx = len(ordered) - 1 if next_idx is None else int(np.floor((idx + next_idx) / 2))
        window = ordered.iloc[start_idx:end_idx + 1]
        time_begin = np.concatenate([np.atleast_1d(v).astype(float) for v in window["Time_begin"]])
        time_end = np.concatenate([np.atleast_1d(v).astype(float) for v in window["Time_end"]])
        selected.append({
            "Potential": float(potentials[idx]),
            "Time_begin": time_begin,
            "Time_end": time_end,
        })
    return selected

def ca_t1_detail_plot_data(detail):
    if detail.get("fit_error") and len(detail.get("raw_qs_tau", [])) and len(detail.get("raw_qs_contrast", [])):
        return detail["raw_qs_tau"], detail["raw_qs_contrast"]
    return detail["qs_tau"], detail["qs_contrast"]

def add_qs_measurement_parameters(info_dic, qs_data_reader, prefix=""):
    params = getattr(qs_data_reader, "params", None)
    if not isinstance(params, dict):
        return

    experiment = params.get("experiment_type", getattr(qs_data_reader, "experiment_type", ""))
    add_if_not_none(info_dic, f"{prefix}Experiment", experiment)
    add_if_not_none(info_dic, f"{prefix}Averages", params.get("averages"))
    add_if_not_none(info_dic, f"{prefix}Dynamic points", params.get("dynamic_steps", get_nested(params, ["data", "dynamic_steps"])))
    add_if_not_none(info_dic, f"{prefix}Averaging mode", get_nested(params, ["data", "averaging_mode"]))
    add_if_not_none(info_dic, f"{prefix}Sensor measurements", get_nested(params, ["sensor", "config", "number_measurements"]))

    if experiment in ("ESR", "ESR_ECQS"):
        min_freq = get_nested(params, ["dynamic_devices", "mw_source", "config", "frequency", 0, 1, 0])
        max_freq = get_nested(params, ["dynamic_devices", "mw_source", "config", "frequency", 0, 1, 1])
        mw_amp = get_nested(params, ["dynamic_devices", "mw_source", "config", "amplitude", 0, 1, 0])
        laser_spot = get_nested(params, ["dynamic_devices", "mw_source", "config", "frequency", 1, 1, 0])
        add_if_not_none(info_dic, f"{prefix}MW frequency min", scale_value(min_freq, 1e-9), "GHz")
        add_if_not_none(info_dic, f"{prefix}MW frequency max", scale_value(max_freq, 1e-9), "GHz")
        add_if_not_none(info_dic, f"{prefix}MW amplitude", mw_amp, "dB")
        add_if_not_none(info_dic, f"{prefix}Laser spot", scale_value(laser_spot, 1e-6), "MHz")
        return

    if experiment in ("Rabi", "Rabi_ECQS"):
        mw_freq = get_nested(params, ["static_devices", "mw_source", "config", "frequency", 0, 1])
        mw_amp = get_nested(params, ["static_devices", "mw_source", "config", "amplitude", 0, 1])
        laser_spot = get_nested(params, ["static_devices", "mw_source", "config", "frequency", 1, 1])
        pi_durations = params.get("pi_pulse_durations", [])
        add_if_not_none(info_dic, f"{prefix}ESR frequency", scale_value(mw_freq, 1e-9), "GHz")
        add_if_not_none(info_dic, f"{prefix}MW amplitude", mw_amp, "dB")
        add_if_not_none(info_dic, f"{prefix}Laser spot", scale_value(laser_spot, 1e-6), "MHz")
        add_if_not_none(info_dic, f"{prefix}Rabi max pulse", safe_max(pi_durations), "ns")
        add_if_not_none(info_dic, f"{prefix}Rabi points", safe_len(pi_durations))
        return

    if experiment in ("T1", "T1_ECQS"):
        mw_freq = get_nested(params, ["static_devices", "mw_source", "config", "frequency", 0, 1])
        mw_amp = get_nested(params, ["static_devices", "mw_source", "config", "amplitude", 0, 1])
        laser_spot = get_nested(params, ["static_devices", "mw_source", "config", "frequency", 1, 1])
        pi_duration = get_nested(params, ["pulse_sequence", "mw_duration"])
        tau_max = get_nested(params, ["pulse_sequence", "t1_max"])
        tau_spacings = params.get("tau_spacings", [])
        add_if_not_none(info_dic, f"{prefix}ESR frequency", scale_value(mw_freq, 1e-9), "GHz")
        add_if_not_none(info_dic, f"{prefix}MW amplitude", mw_amp, "dB")
        add_if_not_none(info_dic, f"{prefix}Laser spot", scale_value(laser_spot, 1e-6), "MHz")
        add_if_not_none(info_dic, f"{prefix}Pi pulse duration", scale_value(pi_duration, 1000), "ns")
        add_if_not_none(info_dic, f"{prefix}T1 max tau", scale_value(tau_max, 0.001), "ms")
        add_if_not_none(info_dic, f"{prefix}T1 spacing", get_nested(params, ["pulse_sequence", "spacing_type"]))
        add_if_not_none(info_dic, f"{prefix}T1 points", safe_len(tau_spacings))

def get_nested(data, path):
    current = data
    try:
        for key in path:
            current = current[key]
        return current
    except (KeyError, IndexError, TypeError):
        return None

def scale_value(value, factor):
    if value is None:
        return None
    try:
        return float(value) * factor
    except (TypeError, ValueError):
        return value

def safe_len(value):
    try:
        return len(value)
    except TypeError:
        return None

def safe_max(value):
    try:
        return max(value) if len(value) else None
    except TypeError:
        return None

def add_if_not_none(info_dic, key, value, unit=None):
    if value is None:
        return
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    info_dic[key] = f"{text} {unit}" if unit else text

def normalize_averaged_decay(mean, std=None, return_cutoff=False):
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    if mean.size == 0:
        if std is None:
            return (mean, 0) if return_cutoff else mean
        normalized_std = np.atleast_1d(np.asarray(std, dtype=float))
        return (mean, normalized_std, 0) if return_cutoff else (mean, normalized_std)

    cutoff_idx = int(np.argmin(mean))
    normalized_mean = 1.0 - mean[cutoff_idx:]
    scale = np.max(normalized_mean)
    if scale == 0:
        normalized_std = None if std is None else np.zeros_like(np.atleast_1d(np.asarray(std, dtype=float))[cutoff_idx:])
    else:
        normalized_mean = normalized_mean / scale
        normalized_std = None if std is None else np.atleast_1d(np.asarray(std, dtype=float))[cutoff_idx:] / abs(scale)

    if std is None:
        return (normalized_mean, cutoff_idx) if return_cutoff else normalized_mean
    return (normalized_mean, normalized_std, cutoff_idx) if return_cutoff else (normalized_mean, normalized_std)

def positive_tau_lower_bound(x_data):
    x_data = np.atleast_1d(np.asarray(x_data, dtype=float))
    positive_x = x_data[x_data > 0]
    if positive_x.size > 0:
        return float(np.min(positive_x))
    return 1e-12

def stretched_exp(x,tau, beta):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.exp(-((x / tau) ** beta))

def bi_exp(x,tau1, tau2, a):
    return a * np.exp(-x / tau1) + (1 - a) * np.exp(-x / tau2)

def ca_time_scale(seconds):
    seconds = np.asarray(seconds, dtype=float)
    if seconds.size and np.nanmax(seconds) >= 1000:
        return seconds / 1000, "Time [ks]"
    return seconds, "Time [s]"

def ca_time_region(fitting_dic, available_end):
    """Return a validated elapsed-time range; a zero end means all remaining data."""
    available_end = max(0.0, float(available_end))
    try:
        time_from = min(max(0.0, float(fitting_dic.get("time_from", 0) or 0)), available_end)
    except (TypeError, ValueError):
        time_from = 0.0
    try:
        requested_to = float(fitting_dic.get("time_to", 0) or 0)
    except (TypeError, ValueError):
        requested_to = 0.0
    time_to = available_end if requested_to <= 0 else min(max(0.0, requested_to), available_end)
    if time_to < time_from:
        time_from, time_to = time_to, time_from
    return time_from, time_to

def ca_callback_time(method, callback_data):
    current = callback_data.get("Current", [])
    callback_time = np.asarray(callback_data.get("Time", []), dtype=float)
    if callback_time.size != len(current):
        interval_time = float(method.get("interval_time", 0.5))
        callback_time = np.arange(len(current), dtype=float) * interval_time
    elif callback_time.size:
        callback_time = callback_time - callback_time[0]
    return callback_time

def ca_callback_xy(method, callback_data):
    current = np.asarray(callback_data.get("Current", []), dtype=float)
    callback_time = ca_callback_time(method, callback_data)
    n = min(len(callback_time), len(current))
    return callback_time[:n], current[:n]

def ca_live_ec_trace(potentiostat, method, raw_ec_data, callback_data):
    if raw_ec_data is not None and not raw_ec_data.empty:
        ec_start = float(np.min(raw_ec_data["Time_begin"]))
        ec_time = np.asarray(raw_ec_data["Time"], dtype=float) - ec_start
        ec_current = np.asarray(raw_ec_data["Current"], dtype=float)
        ec_end = float(np.max(raw_ec_data["Time_end"]))
        return ec_start, ec_end, ec_time, ec_current

    rel_time, current = ca_callback_xy(method, callback_data)
    if len(rel_time) == 0:
        return None, None, rel_time, current

    start_info = potentiostat.get_measurement_start_time()
    if start_info:
        interval_time = float(method.get("interval_time", 0.5))
        ec_start = float(start_info[0]) - float(start_info[1]) * interval_time
    else:
        ec_start = time() - float(rel_time[-1])
    ec_end = ec_start + float(rel_time[-1])
    return ec_start, ec_end, rel_time, current

def ca_t1_windows(qs_data_reader, ec_start, ec_end, avg_per_datapoint, time_bin_size, measurement_done):
    t_begin, t_end, _ = qs_data_reader._load_raw_data()
    if len(t_begin) == 0 or ec_start is None or ec_end is None:
        return [], [], []

    qs_centers = (t_begin + t_end) / 2
    in_ec_run = (t_begin >= ec_start) & (t_end <= ec_end)
    qs_indices = np.flatnonzero(in_ec_run)
    windows = []
    median_times = []
    qs_counts = []
    if time_bin_size > 0:
        edges = np.arange(ec_start, ec_end + time_bin_size, time_bin_size)
        for start, end in zip(edges[:-1], edges[1:]):
            if end > ec_end and not measurement_done:
                break
            mask = (qs_centers >= start) & (qs_centers < end)
            if not np.any(mask):
                continue
            windows.append((start, end))
            median_times.append(float(np.median(qs_centers[mask]) - ec_start))
            qs_counts.append(int(np.sum(mask)))
    else:
        full_count = len(qs_indices) // avg_per_datapoint
        if measurement_done and len(qs_indices) % avg_per_datapoint:
            full_count += 1
        for chunk_number in range(full_count):
            chunk_start = chunk_number * avg_per_datapoint
            chunk = qs_indices[chunk_start:chunk_start + avg_per_datapoint]
            if len(chunk) == 0:
                continue
            windows.append((float(t_begin[chunk[0]]), float(t_end[chunk[-1]])))
            median_times.append(float(np.median(qs_centers[chunk]) - ec_start))
            qs_counts.append(len(chunk))
    return windows, median_times, qs_counts

def setup_ca_t1_axes(fig, cache):
    fig.clear()
    axes = fig.subplots(2, 2)
    ax_bar = axes[1,1].twinx()
    ax_contrast = axes[1,0].twinx()
    cache.axes = axes
    cache.ca_t1_axes = {
        "bar": ax_bar,
        "contrast": ax_contrast,
    }
    cache.ca_t1_artists = {
        "current": axes[0,0].plot([], [], color="tab:blue", label="Current")[0],
        "live_current": axes[0,0].plot([], [], color="tab:green", linestyle="-")[0],
        "t1": axes[0,1].plot([], [], color="blue", marker="x", linestyle="-", label="T1 Time")[0],
        "t1_missing": axes[0,1].scatter(
            [],
            [],
            marker="o",
            facecolors="none",
            edgecolors="blue",
            linewidths=1.5,
            label="T1 Time missing fit",
            zorder=4,
        ),
        "r2": axes[1,1].plot([], [], color="tab:blue", marker="x", linestyle="--", label="R^2 T1")[0],
        "r2_missing": axes[1,1].scatter(
            [],
            [],
            marker="o",
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=1.5,
            label="R^2 T1 missing fit",
            zorder=4,
        ),
        "light": axes[1,0].plot([], [], label="Lightlevel", color="tab:cyan")[0],
        "contrast": ax_contrast.plot([], [], label="Contrast scale", color="tab:purple")[0],
        "bars": [],
    }
    ax_bar.set_ylabel("Averages")
    ax_contrast.set_ylabel("Contrast scale")
    return axes

def update_ca_t1_bars(cache, x_values, heights, width):
    for bar in cache.ca_t1_artists.get("bars", []):
        try:
            bar.remove()
        except ValueError:
            pass
    ax_bar = cache.ca_t1_axes["bar"]
    if len(x_values):
        bars = ax_bar.bar(x_values, heights, width=width, label="QS averages", color="tab:blue", alpha=0.3)
        cache.ca_t1_artists["bars"] = list(bars)
    else:
        cache.ca_t1_artists["bars"] = []

def update_ca_t1_t1_artists(cache, x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(y_values)
    missing = ~valid
    cache.ca_t1_artists["t1"].set_data(x_values[valid], y_values[valid])
    if np.any(missing) and np.any(valid):
        sort_idx = np.argsort(x_values[valid])
        marker_y = np.interp(
            x_values[missing],
            x_values[valid][sort_idx],
            y_values[valid][sort_idx],
        )
        cache.ca_t1_artists["t1_missing"].set_offsets(np.column_stack((x_values[missing], marker_y)))
    else:
        cache.ca_t1_artists["t1_missing"].set_offsets(np.empty((0, 2)))

def update_ca_t1_r2_artists(cache, x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(y_values)
    missing = ~valid
    cache.ca_t1_artists["r2"].set_data(x_values[valid], y_values[valid])
    if np.any(missing) and np.any(valid):
        sort_idx = np.argsort(x_values[valid])
        marker_y = np.interp(
            x_values[missing],
            x_values[valid][sort_idx],
            y_values[valid][sort_idx],
        )
        cache.ca_t1_artists["r2_missing"].set_offsets(np.column_stack((x_values[missing], marker_y)))
    else:
        cache.ca_t1_artists["r2_missing"].set_offsets(np.empty((0, 2)))

def ca_t1_progress(method, ec_time, measurement_done):
    if measurement_done:
        return 1
    run_time = float(method.get("run_time", 0) or 0)
    if run_time <= 0 or len(ec_time) == 0:
        return 0
    return max(0, min(float(np.nanmax(ec_time)) / run_time, 1))

def fit_t1_from_qs_window(qs_data_reader, time_window, cutoff_rate, fitting_dic, maxfev, use_r_square_threshold, r_square_threshold, return_detail=False):
    qs_data = qs_data_reader.evaluate_qs_data(cutoff_rate=cutoff_rate, time_array=np.asarray([time_window]))
    qs_tau = qs_data["x_data"]
    qs_contrast = qs_data["contrast_mean"]
    contrast_scale = np.nan
    detail = {
        "raw_qs_tau": np.array([]),
        "raw_qs_contrast": np.array([]),
        "qs_tau": np.array([]),
        "qs_contrast": np.array([]),
        "y_fit": None,
        "tau": None,
        "r_square": None,
        "fit_error": None,
    }
    try:
        detail["raw_qs_tau"] = np.atleast_1d(np.asarray(qs_tau, dtype=float))
        detail["raw_qs_contrast"] = np.atleast_1d(np.asarray(qs_contrast, dtype=float))
        if len(detail["raw_qs_contrast"]) == 0 and len(qs_data["reference_mean"]) and len(qs_data["value_mean"]):
            raw_reference = np.atleast_1d(np.asarray(qs_data["reference_mean"], dtype=float))
            raw_value = np.atleast_1d(np.asarray(qs_data["value_mean"], dtype=float))
            detail["raw_qs_contrast"] = np.divide(
                raw_value,
                raw_reference,
                out=np.zeros_like(raw_value, dtype=float),
                where=raw_reference != 0,
            )
        detail["qs_tau"] = detail["raw_qs_tau"]
        detail["qs_contrast"] = detail["raw_qs_contrast"]
    except (TypeError, ValueError):
        pass
    try:
        qs_tau = np.atleast_1d(np.asarray(qs_tau, dtype=float))
        raw_qs_contrast = np.atleast_1d(np.asarray(qs_contrast, dtype=float))
        qs_contrast, t1_cutoff_idx = normalize_averaged_decay(qs_contrast, return_cutoff=True)
        qs_tau = qs_tau[t1_cutoff_idx:]
        detail["qs_tau"] = qs_tau
        detail["qs_contrast"] = qs_contrast
        contrast_window = 1.0 - raw_qs_contrast[t1_cutoff_idx:]
        contrast_scale = float(np.max(contrast_window)) if contrast_window.size else np.nan
    except (TypeError, ValueError):
        detail["fit_error"] = "data could not be normalized"
        if return_detail:
            return np.nan, np.nan, 0, np.nan, contrast_scale, detail
        return np.nan, np.nan, 0, np.nan, contrast_scale

    cl = qs_data["contrast_length"]
    if isinstance(cl, (list, np.ndarray)) and len(cl) > 0:
        qs_points = cl[0]
    elif isinstance(cl, (int, float)):
        qs_points = cl
    else:
        qs_points = 0
    qs_lightlevel = np.mean(qs_data["reference_mean"]) if len(qs_data["reference_mean"]) else np.nan
    if len(qs_contrast) < 3 or len(qs_tau) < 3:
        detail["fit_error"] = "not enough data"
        if return_detail:
            return np.nan, np.nan, qs_points, qs_lightlevel, contrast_scale, detail
        return np.nan, np.nan, qs_points, qs_lightlevel, contrast_scale

    min_tau = positive_tau_lower_bound(qs_tau)
    p0 = [qs_tau[-1] / 3, 1.0]
    if "use_bounds" in fitting_dic.keys() and fitting_dic["use_bounds"] == True:
        bounds = ([min_tau, 0.1], [qs_tau[-1] * 10, 3.0])
    else:
        bounds = ([min_tau, 0.05], [max(qs_tau[-1] * 1000, min_tau * 10), 10.0])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(stretched_exp, qs_tau, qs_contrast, p0=p0, bounds=bounds, maxfev=maxfev)
        tau = popt[0]
        if tau <= 0 or popt[1] <= 0:
            raise ValueError(f"Invalid T1 fit parameters: tau={tau}, beta={popt[1]}")
        y_fit = stretched_exp(qs_tau, *popt)
        if not np.all(np.isfinite(y_fit)):
            raise ValueError("T1 fit produced non-finite values")
        r_square = r_squared(qs_contrast, y_fit)
        if use_r_square_threshold and r_square < r_square_threshold:
            detail["fit_error"] = f"R^2 below threshold: {r_square:.3f} < {r_square_threshold:.3f}"
            if return_detail:
                return np.nan, np.nan, qs_points, qs_lightlevel, contrast_scale, detail
            return np.nan, np.nan, qs_points, qs_lightlevel, contrast_scale
        detail["y_fit"] = y_fit
        detail["tau"] = tau
        detail["r_square"] = r_square
        if return_detail:
            return tau, r_square, qs_points, qs_lightlevel, contrast_scale, detail
        return tau, r_square, qs_points, qs_lightlevel, contrast_scale
    except Exception as e:
        detail["fit_error"] = str(e)
        if return_detail:
            return np.nan, np.nan, qs_points, qs_lightlevel, contrast_scale, detail
        return np.nan, np.nan, qs_points, qs_lightlevel, contrast_scale



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
    elif method["id"] == "ad":
        info_dic["Potential"] = method["potential"]
        info_dic["Run time"] = method["run_time"]
        info_dic["Interval time"] = method["interval_time"]
    #print(method)
    #info_dic["Method"] = method["General"]
    current_measurement = potentiostat.get_measurement_count()
    n_measurements = potentiostat.n_scans
    max_points = potentiostat.max_points
    info_dic["Final Length"] = f"{n_measurements} x {max_points}"
    callback_data = potentiostat.get_callback_data()
    #print(callback_data)
    callback_length = len(callback_data.get("Current", [])) if method["id"] == "ad" else len(callback_data["Potential"])
    info_dic["Current Length"] = f"{current_measurement} x {callback_length}"
    measurement_progress = (current_measurement*max_points + callback_length) / max(n_measurements*max_points, 1)
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
    if method["id"] == "ad":
        fig.clear()
        axes = fig.subplots()
        try:
            if not measurement_done:
                callback_time, callback_current = ca_callback_xy(method, callback_data)
                axes.plot(callback_time, callback_current, color="tab:green", linestyle="-")
        except Exception:
            pass
        if data is not None:
            axes.plot(data["Relative_time"], data["Current"], color="tab:blue", label="Current")
        available_end = 0.0
        if data is not None and len(data):
            available_end = max(available_end, float(np.nanmax(data["Relative_time"])))
        callback_time, _ = ca_callback_xy(method, callback_data)
        if len(callback_time):
            available_end = max(available_end, float(np.nanmax(callback_time)))
        time_from, time_to = ca_time_region(fitting_dic, available_end)
        if time_to > time_from:
            axes.set_xlim(time_from, time_to)
        info_dic["Displayed time region"] = f"{time_from:g} - {time_to:g} s"
        axes.set_xlabel("Time [s]")
        axes.set_ylabel("Current [uA]")
        axes.grid(True, alpha=0.3)
        combined_legend(axes)
        return fig, axes, measurement_done, measurement_progress, info_dic
    #Plot Callback
    try:
        axes[0].plot(callback_data['Potential'],callback_data["Current"],"gx")
    except:
        pass
    
    axes[0].set_xlabel("Potential [V] vs SSC")
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
    axes[1].plot(range(1, len(integral) + 1), integral, color="tab:orange", label="Current sum")
    axes[1].set_xlabel("N Measurement")
    axes[1].set_ylabel("Abs Current Sum [A]")
    combined_legend(axes[1])
    return fig, axes, measurement_done, measurement_progress, info_dic

def ca_t1_data_displayer(qs_data_reader, potentiostat, fitting_dic: dict, cache, fig: Figure | None = None):
    cutoff_rate = fitting_dic.get("cutoff_rate", 0.5)
    maxfev = fitting_dic.get("max_tries", 5000)
    use_r_square_threshold = bool(fitting_dic.get("use_r_square_threshold", False))
    try:
        r_square_threshold = float(fitting_dic.get("r_square_threshold", 0.95))
    except (TypeError, ValueError):
        r_square_threshold = 0.95
    try:
        avg_per_datapoint = max(1, int(fitting_dic.get(
            "t1_averages_per_datapoint",
            getattr(potentiostat, "avg_per_point", 300) or 300,
        )))
    except (TypeError, ValueError):
        avg_per_datapoint = 300
    try:
        time_bin_size = float(fitting_dic.get("time_bin_size", 0))
    except (TypeError, ValueError):
        time_bin_size = 0
    save_detail_plots = bool(fitting_dic.get("save_detail_plots", False))

    method = potentiostat.method.to_dict()
    raw_ec_data = potentiostat.get_measurement_dataframe()
    callback_data = potentiostat.get_callback_data()
    qs_data = qs_data_reader.evaluate_qs_data()
    time_label = "Time [s]"
    info_dic = {
        "Potential": method["potential"],
        "Run time": method["run_time"],
        "Interval time": method["interval_time"],
        "T1 averages/datapoint": avg_per_datapoint,
    }
    if time_bin_size > 0:
        info_dic["Time bin size"] = f"{time_bin_size} s"
    if use_r_square_threshold:
        info_dic["T1 R^2 threshold"] = f"{r_square_threshold:.3f}"
    try:
        info_dic["QS:Measurement_time"] = f"{np.mean(qs_data['time_measure_mean'])*1000:.0f} +- {np.mean(qs_data['time_measure_std'])*1000:.0f} ms"
        info_dic["QS:Saving time"] = f"{np.mean(qs_data['time_save_mean'])*1000:.0f} +- {np.mean(qs_data['time_save_std'])*1000:.0f} ms"
        info_dic["QS:Light Level"] = f"{np.mean(qs_data['reference_mean']):.4f} mV"
        info_dic["QS:Measurement points"] = f"{len(qs_data['x_data'])} points x {qs_data['contrast_length'][0]} averages"
        add_qs_measurement_parameters(info_dic, qs_data_reader, prefix="QS:")
    except Exception:
        pass

    measurement_done = False
    kill_qupyt = False
    if potentiostat.potentiostat_disconnected.is_set() and qs_data_reader.dataframe_loop_stop.is_set():
        measurement_done=True
    elif potentiostat.potentiostat_disconnected.is_set():
        kill_qupyt = True
    if fig is None:
        fig = Figure(figsize=(12,8), constrained_layout=True)
    ec_start, ec_end, ec_time, ec_current = ca_live_ec_trace(potentiostat, method, raw_ec_data, callback_data)
    windows, median_times, qs_counts = ca_t1_windows(
        qs_data_reader,
        ec_start,
        ec_end,
        avg_per_datapoint,
        time_bin_size,
        measurement_done,
    )
    available_end = float(np.nanmax(ec_time)) if len(ec_time) else float(method.get("run_time", 0) or 0)
    time_from, time_to = ca_time_region(fitting_dic, available_end)
    selected = [i for i, value in enumerate(median_times) if time_from <= value <= time_to]
    windows = [windows[i] for i in selected]
    median_times = [median_times[i] for i in selected]
    qs_counts = [qs_counts[i] for i in selected]
    info_dic["Displayed time region"] = f"{time_from:g} - {time_to:g} s"
    completed_points = len(windows)
    measurement_progress = ca_t1_progress(method, ec_time, measurement_done)
    if cache.axes is None or cache.fig is None:
        axes = setup_ca_t1_axes(fig, cache)
        cache.fig = fig
    else:
        axes = cache.axes
        fig = cache.fig

    scaled_ec_time, time_label = ca_time_scale(ec_time)
    cache.ca_t1_artists["current"].set_data(scaled_ec_time, ec_current)
    if not measurement_done:
        callback_time, callback_current = ca_callback_xy(method, callback_data)
        scaled_callback_time, _ = ca_time_scale(callback_time)
        cache.ca_t1_artists["live_current"].set_data(scaled_callback_time, callback_current)
    else:
        cache.ca_t1_artists["live_current"].set_data([], [])

    if ec_start is not None and (completed_points > cache.last_measurement or measurement_done):
        sub_path = os.path.join(qs_data_reader.file_path, qs_data_reader.file_name)
        if save_detail_plots:
            os.makedirs(sub_path, exist_ok=True)
            clear_ca_t1_detail_plots(sub_path)
        tau_values = []
        r_square_values = []
        light_values = []
        contrast_values = []
        for point_number, window in enumerate(windows, start=1):
            fit_result = fit_t1_from_qs_window(
                qs_data_reader,
                window,
                cutoff_rate,
                fitting_dic,
                maxfev,
                use_r_square_threshold,
                r_square_threshold,
                return_detail=save_detail_plots,
            )
            if save_detail_plots:
                tau, r_square, _, lightlevel, contrast_scale, detail = fit_result
                detail_tau, detail_contrast = ca_t1_detail_plot_data(detail)
                save_ca_t1_detail_plot(
                    point_number,
                    detail_tau,
                    detail_contrast,
                    sub_path,
                    y_fit=detail["y_fit"],
                    tau=detail["tau"],
                    r_square=detail["r_square"],
                    fit_error=detail["fit_error"],
                )
            else:
                tau, r_square, _, lightlevel, contrast_scale = fit_result
            tau_values.append(tau)
            r_square_values.append(r_square)
            light_values.append(lightlevel)
            contrast_values.append(contrast_scale)

        scaled_median_times, time_label = ca_time_scale(median_times)
        cache.ca_t1_artists["light"].set_data(scaled_median_times, light_values)
        cache.ca_t1_artists["contrast"].set_data(scaled_median_times, contrast_values)
        if len(scaled_median_times):
            if len(scaled_median_times) > 1:
                width = np.nanmedian(np.diff(np.asarray(scaled_median_times, dtype=float))) * 0.8
            elif time_bin_size > 0:
                width = time_bin_size / (1000 if "ks" in time_label else 1) * 0.8
            else:
                width = 0.1
        else:
            width = 0.1
        update_ca_t1_t1_artists(cache, scaled_median_times, tau_values)
        update_ca_t1_r2_artists(cache, scaled_median_times, r_square_values)
        update_ca_t1_bars(cache, scaled_median_times, qs_counts, width)

        cache.last_measurement = completed_points
        cache.axes = axes
        cache.measurement_progress = measurement_progress
        cache.kill_qupyt = kill_qupyt
        cache.fig = fig
    else:
        cache.measurement_progress = measurement_progress

    fig.suptitle("Chronoamperometry / T1_ECQS")
    axes[0,0].set_xlabel(time_label)
    axes[0,0].set_ylabel("Current [uA]")
    axes[0,1].set_xlabel(time_label)
    axes[0,1].set_ylabel("T1 Time")
    set_t1_axis_from_zero(axes[0,1])
    axes[1,0].set_xlabel(time_label)
    axes[1,0].set_ylabel("Lightlevel [mV]")
    axes[1,1].set_xlabel(time_label)
    axes[1,1].set_ylabel("R^2")
    axes[1,1].set_ylim(0, 1.1)
    cache.ca_t1_axes["bar"].relim()
    cache.ca_t1_axes["bar"].autoscale_view()
    cache.ca_t1_axes["contrast"].relim()
    cache.ca_t1_axes["contrast"].autoscale_view()
    for ax in axes.flat:
        ax.relim()
        ax.autoscale_view()
        ax.grid(True, alpha=0.3)
        combined_legend(ax, loc="upper left")
        scaled_from, _ = ca_time_scale(time_from)
        scaled_to, _ = ca_time_scale(time_to)
        if scaled_to > scaled_from:
            ax.set_xlim(scaled_from, scaled_to)
    return fig, axes, measurement_done, measurement_progress, info_dic, kill_qupyt
    
def ecqs_data_displayer(qs_data_reader, potentiostat, fitting_dic: dict, cache, fig: Figure | None = None):
    measurement_count = potentiostat.get_measurement_count()
    if measurement_count:
        measurement_from, measurement_to = potentiostat.normalize_measurement_range(
            fitting_dic.get("measurement_from", 1),
            fitting_dic.get("measurement_to", 0),
        )
    else:
        # Keep the live view usable while the first EC scan is still running.
        measurement_from, measurement_to = 1, 0
    if "cutoff_rate" in fitting_dic.keys():
        cutoff_rate = fitting_dic["cutoff_rate"]
    else:
        cutoff_rate = 0.5
    if "max_tries" in fitting_dic.keys():
         maxfev = fitting_dic["max_tries"]
    else:
         maxfev = 50
    use_r_square_threshold = bool(fitting_dic.get("use_r_square_threshold", False))
    try:
        r_square_threshold = float(fitting_dic.get("r_square_threshold", 0.95))
    except (TypeError, ValueError):
        r_square_threshold = 0.95
    try:
        t1_potential_step = abs(float(fitting_dic.get("t1_potential_step", 0.05)))
    except (TypeError, ValueError):
        t1_potential_step = 0.05
    if not np.isfinite(t1_potential_step) or t1_potential_step <= 0:
        t1_potential_step = 0.05
        
    info_dic ={}
    info_dic["Included measurements"] = f"{measurement_from}-{measurement_to} of {measurement_count}"
    info_dic["T1 Potential Step"] = f"{t1_potential_step * 1000:.0f} mV"
    if use_r_square_threshold:
        info_dic["T1 R^2 threshold"] = f"{r_square_threshold:.3f}"
    measurement_done = False
    kill_qupyt =False
    method = potentiostat.method.to_dict()
    if method["id"] == "ad":
        return ca_t1_data_displayer(qs_data_reader, potentiostat, fitting_dic, cache, fig)
    qs_data = qs_data_reader.evaluate_qs_data()
    callback_data = potentiostat.get_callback_data()
   
    current_measurement = measurement_count
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
        info_dic["QS:Light Level"] = f"{np.mean(qs_data['reference_mean']):.4f} mV"
        info_dic["QS:Measurement points"] = f"{len(qs_data['x_data'])} points x {qs_data['contrast_length'][0]} averages"
        add_qs_measurement_parameters(info_dic, qs_data_reader, prefix="QS:")
    except:
        pass
    
    save_detail_plots = bool(fitting_dic.get("fitting", False)) or "ec_fitting" in fitting_dic
    sub_path = os.path.join(qs_data_reader.file_path, qs_data_reader.file_name)
    if save_detail_plots:
        os.makedirs(sub_path,exist_ok=True)
    all_up_fig = Figure(figsize=(12,8), constrained_layout=True)
    all_up_ax = all_up_fig.subplots()
    all_down_fig = Figure(figsize=(12,8), constrained_layout=True)
    all_down_ax = all_down_fig.subplots()
    
    ec_data = potentiostat.evaluated_data(measurement_from, measurement_to)
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
        ax_contrast_with_pot = axes[1,1].twinx()
        ax_contrast_with_pot.spines["right"].set_position(("outward", 120))
        ax_contrast_with_pot.set_ylabel("Contrast scale")
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
                t1_points_up = ecqs_t1_selected_points(plot_up, t1_potential_step)
                x_values_up = np.asarray([point["Potential"] for point in t1_points_up], dtype=float)
                qs_light_up = []
                contrast_scale_up = []
                print(f"v_values_up length: {len(x_values_up)}")
                all_up_ax.set_prop_cycle(color=mpl.colormaps['viridis'](np.linspace(0.2, 1.0, max(len(t1_points_up), 1))))
                print("Processing plot up")
                for row in tqdm(t1_points_up):
                    time_begin = np.atleast_1d(row["Time_begin"])
                    time_end   = np.atleast_1d(row["Time_end"])
                    time_frames_up = np.column_stack((time_begin,time_end))
                   # print("shape:", time_frames_up.shape)
                    #print(time_frames_up)
                    qs_data_up = qs_data_reader.evaluate_qs_data(cutoff_rate=cutoff_rate,time_array=time_frames_up)
                    qs_tau = qs_data_up["x_data"]
                    qs_contrast = qs_data_up["contrast_mean"]
                    qs_lightlevel = np.mean(qs_data_up["reference_mean"])
                    contrast_scale = np.nan
                    
                    try:
                        qs_tau = np.atleast_1d(np.asarray(qs_tau, dtype=float))
                        raw_qs_contrast = np.atleast_1d(np.asarray(qs_contrast, dtype=float))
                        qs_contrast, t1_cutoff_idx = normalize_averaged_decay(qs_contrast, return_cutoff=True)
                        qs_tau = qs_tau[t1_cutoff_idx:]
                        contrast_window = 1.0 - raw_qs_contrast[t1_cutoff_idx:]
                        contrast_scale = float(np.max(contrast_window)) if contrast_window.size else np.nan
                    except (TypeError, ValueError):
                        qs_tau = np.array([])
                        qs_contrast = np.array([])
                        print("data could not be normalized")
                    cl = qs_data_up["contrast_length"]
                    if isinstance(cl, (list, np.ndarray)) and len(cl) > 0:
                        qs_points = cl[0]
                    elif isinstance(cl, (int, float)):
                        qs_points = cl
                    else:
                        qs_points = 0
                    if len(qs_contrast) <3 or len(qs_tau) <3:
                        if save_detail_plots:
                            save_ecqs_t1_detail_plot(
                                qs_tau,
                                qs_contrast,
                                float(row["Potential"]),
                                "blue",
                                "Up",
                                sub_path,
                                all_up_ax,
                                fit_error="not enough data",
                            )
                        tau_values_up.append(np.nan)
                        r_square_up.append(np.nan)
                        points_up.append(qs_points)
                        qs_light_up.append(qs_lightlevel)
                        contrast_scale_up.append(contrast_scale)
                        continue
                    #print("Contrast length:", len(qs_contrast))
                   # print(len(qs_tau))
                    
                    min_tau = positive_tau_lower_bound(qs_tau)
                    p0 = [qs_tau[-1] / 3,  # tau ~ 1/3 of x range
                          1.0              # beta
                          ]
                    if "use_bounds" in fitting_dic.keys() and fitting_dic["use_bounds"]==True:
                        bounds = (
                            [min_tau,          0.1],          # lower
                            [qs_tau[-1] * 10,  3.0]           # upper
                        )
                    else:
                        bounds = (-np.inf, np.inf)
                    #time_0=time()
                    y_fit = None
                    tau = None
                    r_square = None
                    fit_error = None
                    try:
                        popt, pcov = curve_fit(stretched_exp, qs_tau, qs_contrast, p0=p0, bounds=bounds,maxfev=maxfev)
                        tau = popt[0]
                        if tau < 0:
                            raise ValueError(f"Negative T1 fit tau: {tau}")
                        y_fit = stretched_exp(qs_tau, *popt)
                        r_square = r_squared(qs_contrast, y_fit)
                        if use_r_square_threshold and r_square < r_square_threshold:
                            failed_r_square = r_square
                            y_fit = None
                            tau = None
                            r_square = None
                            raise ValueError(f"R^2 below threshold: {failed_r_square:.3f} < {r_square_threshold:.3f}")
                        tau_values_up.append(tau)
                        r_square_up.append(r_square)
                        """
                        Plot figure for every potential
                        """
                    except Exception as e:
                        #print(e)
                        fit_error = str(e)
                        tau_values_up.append(np.nan)
                        r_square_up.append(np.nan)
                    finally:
                        if save_detail_plots:
                            save_ecqs_t1_detail_plot(
                                qs_tau,
                                qs_contrast,
                                float(row["Potential"]),
                                "blue",
                                "Up",
                                sub_path,
                                all_up_ax,
                                y_fit=y_fit,
                                tau=tau,
                                r_square=r_square,
                                fit_error=fit_error,
                            )
                        points_up.append(qs_points)
                        qs_light_up.append(qs_lightlevel)
                        contrast_scale_up.append(contrast_scale)
                  #  print(f"Up-fittingtime:{(time()-time_0)*1000} ms")
                try:
                    plot_line_with_missing_fit_markers(
                        axes[0,1],
                        x_values_up,
                        tau_values_up,
                        "blue",
                        "T1 Time Upwards current",
                    )
                except Exception as e:
                    print(e)
                try:
                    x_up_array = np.asarray(x_values_up, dtype=float)
                    r_square_up_array = np.asarray(r_square_up, dtype=float)
                    valid_r_square_up = np.isfinite(r_square_up_array)
                    axes[1,1].plot(x_up_array[valid_r_square_up], r_square_up_array[valid_r_square_up], color="blue", linestyle="--", label="R^2 T1")
                except Exception as e:
                    print(e)
                try:
                    ax_light_with_pot.plot(x_values_up, qs_light_up, color="tab:blue", label = "Up Lightlevel")
                    ax_contrast_with_pot.plot(x_values_up, contrast_scale_up, color="tab:purple", linestyle="-", label="Up Contrast scale")
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
                t1_points_down = ecqs_t1_selected_points(plot_down, t1_potential_step)
                x_values_down = np.asarray([point["Potential"] for point in t1_points_down], dtype=float)
                qs_light_down = []
                contrast_scale_down = []
                print(f"v_values_down length: {len(x_values_down)}")
                all_down_ax.set_prop_cycle(color=mpl.colormaps['viridis'](np.linspace(0.2, 1.0, max(len(t1_points_down), 1))))
                print("Processing plot down")
                for row in tqdm(t1_points_down):
                    time_begin = np.atleast_1d(row["Time_begin"])
                    time_end   = np.atleast_1d(row["Time_end"])
                    time_frames_down = np.column_stack((time_begin,time_end))
                   # print("shape:", time_frames_down.shape)
                    qs_data_down = qs_data_reader.evaluate_qs_data(cutoff_rate=cutoff_rate,time_array=time_frames_down)
                    qs_tau = qs_data_down["x_data"]
                    qs_contrast = qs_data_down["contrast_mean"]
                    qs_lightlevel = np.mean(qs_data_down["reference_mean"])
                    contrast_scale = np.nan
                    try:
                        qs_tau = np.atleast_1d(np.asarray(qs_tau, dtype=float))
                        raw_qs_contrast = np.atleast_1d(np.asarray(qs_contrast, dtype=float))
                        qs_contrast, t1_cutoff_idx = normalize_averaged_decay(qs_contrast, return_cutoff=True)
                        qs_tau = qs_tau[t1_cutoff_idx:]
                        contrast_window = 1.0 - raw_qs_contrast[t1_cutoff_idx:]
                        contrast_scale = float(np.max(contrast_window)) if contrast_window.size else np.nan
                    except (TypeError, ValueError):
                        qs_tau = np.array([])
                        qs_contrast = np.array([])
                        print("data could not be normalized")
                    cl = qs_data_down["contrast_length"]
                    if isinstance(cl, (list, np.ndarray)) and len(cl) > 0:
                        qs_points = cl[0]
                    elif isinstance(cl, (int, float)):
                        qs_points = cl
                    else:
                        qs_points = 0
                    if len(qs_contrast) < 3 or len(qs_tau) < 3:
                        if save_detail_plots:
                            save_ecqs_t1_detail_plot(
                                qs_tau,
                                qs_contrast,
                                float(row["Potential"]),
                                "red",
                                "Down",
                                sub_path,
                                all_down_ax,
                                fit_error="not enough data",
                            )
                        tau_values_down.append(np.nan)
                        r_square_down.append(np.nan)
                        points_down.append(qs_points)
                        qs_light_down.append(qs_lightlevel)
                        contrast_scale_down.append(contrast_scale)
                        continue
                   # print("Contrast length: ", len(qs_contrast))
                   # print(len(qs_tau))
                       
                    
                    min_tau = positive_tau_lower_bound(qs_tau)
                    p0 = [qs_tau[-1] / 3,  # tau ~ 1/3 of x range
                          1.0              # beta
                          ]
                    if "use_bounds" in fitting_dic.keys() and fitting_dic["use_bounds"]==True:
                        bounds = (
                            [min_tau,          0.1],          # lower
                            [qs_tau[-1] * 10,  3.0]           # upper
                        )
                    else:
                        bounds = (-np.inf, np.inf)
                   # t0 = time()
                    y_fit = None
                    tau = None
                    r_square = None
                    fit_error = None
                    try:
                        popt, pcov = curve_fit(stretched_exp, qs_tau, qs_contrast, p0=p0,bounds=bounds, maxfev=maxfev)
                        tau = popt[0]
                        if tau < 0:
                            raise ValueError(f"Negative T1 fit tau: {tau}")
                        y_fit = stretched_exp(qs_tau, *popt)
                        r_square = r_squared(qs_contrast, y_fit)
                        if use_r_square_threshold and r_square < r_square_threshold:
                            failed_r_square = r_square
                            y_fit = None
                            tau = None
                            r_square = None
                            raise ValueError(f"R^2 below threshold: {failed_r_square:.3f} < {r_square_threshold:.3f}")
                        tau_values_down.append(tau)
                        r_square_down.append(r_square)
                        """
                        Plot figure for every potential
                        """
                        
                    except Exception as e:
                        #print(e)
                        fit_error = str(e)
                        tau_values_down.append(np.nan)
                        r_square_down.append(np.nan)
                    finally:
                        if save_detail_plots:
                            save_ecqs_t1_detail_plot(
                                qs_tau,
                                qs_contrast,
                                float(row["Potential"]),
                                "red",
                                "Down",
                                sub_path,
                                all_down_ax,
                                y_fit=y_fit,
                                tau=tau,
                                r_square=r_square,
                                fit_error=fit_error,
                            )
                        points_down.append(qs_points)
                        qs_light_down.append(qs_lightlevel)
                        contrast_scale_down.append(contrast_scale)
                #print(f"fitting_time: {(time()-t0)*1000} ms")
                try:
                    plot_line_with_missing_fit_markers(
                        axes[0,1],
                        x_values_down,
                        tau_values_down,
                        "red",
                        "T1 Time Downwards current",
                    )
                except Exception as e:
                    print(e)
                try:
                    x_down_array = np.asarray(x_values_down, dtype=float)
                    r_square_down_array = np.asarray(r_square_down, dtype=float)
                    valid_r_square_down = np.isfinite(r_square_down_array)
                    axes[1,1].plot(x_down_array[valid_r_square_down], r_square_down_array[valid_r_square_down], color="red", linestyle="--", label="R^2 T1")
                except Exception as e:
                    print(e)
                try:
                    
                    ax_light_with_pot.plot(x_values_down, qs_light_down, color="tab:red", linestyle="-", label = "Down Lightlevel")
                    ax_contrast_with_pot.plot(x_values_down, contrast_scale_down, color="tab:pink", linestyle="-", label="Down Contrast scale")
                except Exception as e:
                    print(e)
        except Exception as e:
            print(f"Something in plot_down failed {e}")
        axes[1,1].set_ylim(0, 1.1)  # R² is always 0-1
        axes[1,1].set_ylabel("R² / [V:100]")
        axes[1,1].set_xlabel("Potential [V] vs SSC")
        
        # For the twin axis, set its own scale separately:
        ax_bar = axes[1,1].twinx()
        ax_bar.set_ylabel("Averages")
        try:
            ax_bar.bar(x_values_up - t1_potential_step/4, points_up, t1_potential_step/2, label="Upwards Averages",color="blue",alpha=0.3)
        finally:
            try:
                ax_bar.bar(x_values_down + t1_potential_step/4, points_down, t1_potential_step/2, label="Downwards Averages",color="red",alpha=0.3)
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
                
        integral = potentiostat.get_measurement_integrals(measurement_from, measurement_to)
        measurement_numbers = range(measurement_from, measurement_to + 1)
        axes[1,0].plot(measurement_numbers, integral, color="tab:orange", label="Current sum")
        
        #Get all EC figures
        if save_detail_plots:
            try:
                print("Saving EC plots")
                for i,figure in tqdm(enumerate(
                    potentiostat.get_measurement_plots(measurement_from, measurement_to),
                    start=measurement_from,
                )):
                    try:
                        fig_path = os.path.join(sub_path,f"EC_{i}.png")
                        if not os.path.exists(fig_path):
                            figure.savefig(fig_path, dpi=300)
                        export_figure_data_csv(figure, os.path.splitext(fig_path)[0] + ".csv")
                    finally:
                        plt.close(figure)
            except:
                pass
            
        up_fig_name = "QS_Up_all.png"
        down_fig_name = "QS_Down_all.png"
        
        if save_detail_plots:
            all_up_ax.legend(loc="lower right")
            all_down_ax.legend(loc="lower right")
            all_up_fig.savefig(os.path.join(sub_path, up_fig_name), dpi=300)
            all_down_fig.savefig(os.path.join(sub_path, down_fig_name), dpi=300)
            export_figure_data_csv(all_up_fig, os.path.join(sub_path, "QS_Up_all.csv"))
            export_figure_data_csv(all_down_fig, os.path.join(sub_path, "QS_Down_all.csv"))
        plt.close(all_up_fig)
        plt.close(all_down_fig)
        #Code lightlevel over time here
        raw_ec_data = potentiostat.get_measurement_dataframe(measurement_from, measurement_to)
        time_min = np.min(raw_ec_data["Time_begin"])
        time_max = np.max(raw_ec_data["Time_end"])
        selected_measurement_count = measurement_to - measurement_from + 1
        time_lists = np.linspace(time_min, time_max, selected_measurement_count*2)
        x_list = np.linspace(measurement_from, measurement_to, selected_measurement_count*2)
        ax_ll = axes[1,0].twinx()
        ax_ll.set_ylabel("Lightlevel [mV]")
        print("Evaluating Light-level")
        ll_list = lightlevel_by_time_bins(qs_data_reader, time_lists)
        ax_ll.plot(x_list[1:],ll_list, label="Lightlevel",color="tab:cyan")
        ax_contrast_time = axes[1,0].twinx()
        ax_contrast_time.spines["right"].set_position(("outward", 60))
        ax_contrast_time.set_ylabel("Contrast scale")
        print("Evaluating Contrast")
        contrast_time_list = contrast_scale_by_time_bins(qs_data_reader, time_lists)
        ax_contrast_time.plot(x_list[1:], contrast_time_list, label="Contrast scale", color="tab:purple")
        try:
            print("Evaluating T1 time over time")
            t1_time_values, t1_tau_values, t1_points_by_bin = t1_time_by_average_bins(
                qs_data_reader,
                averages_per_bin=3000,
                fitting_dic=fitting_dic,
                maxfev=maxfev,
            )
            t1_time_values, t1_tau_values, t1_points_by_bin = trim_t1_time_bins(
                t1_time_values,
                t1_tau_values,
                t1_points_by_bin,
                time_min,
                time_max,
            )
            if len(t1_time_values) > 0 and time_max > time_min:
                t1_x_values = measurement_from + (
                    (t1_time_values - time_min) / (time_max - time_min)
                ) * (measurement_to - measurement_from)
                ax_t1_time = axes[1,0].twinx()
                ax_t1_time.spines["right"].set_position(("outward", 120))
                ax_t1_time.set_ylabel("T1 Time [ms]")
                plot_line_with_missing_fit_markers(
                    ax_t1_time,
                    t1_x_values,
                    t1_tau_values,
                    "tab:green",
                    "T1 time / 3000 avg",
                )
                set_t1_axis_from_zero(ax_t1_time)
                info_dic["T1 over time bin"] = "3000 averages"
                info_dic["T1 over time bins"] = len(t1_points_by_bin)
        except Exception as e:
            print(f"Evaluating T1 time over time failed: {e}")
        del ll_list,contrast_time_list,x_list
        
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
        print("Evaluation Run complete")
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
        old_callback_artist = getattr(cache, "callback_artist", None)
        if old_callback_artist is not None:
            try:
                old_callback_artist.remove()
            except ValueError:
                pass
        callback_artists = axes[0,0].plot(callback_data['Potential'],callback_data["Current"],"gx")
        cache.callback_artist = callback_artists[0] if callback_artists else None
    except Exception:
        cache.callback_artist = None
    axes[0,0].set_xlabel("Potential [V] vs SSC")
    axes[0,0].set_ylabel("Current [uA]")
    axes[0,1].set_xlabel("Potential [V] vs SSC")
    axes[0,1].set_ylabel("T1 Time")
    set_t1_axis_from_zero(axes[0,1])
    axes[0,0].grid(True, alpha=0.3)
    combined_legend(axes[0,0])
    axes[0,1].grid(True, alpha=0.3)
    combined_legend(axes[0,1])
    axes[1,1].grid(True, alpha=0.3)
    combined_legend(axes[1,1], loc="upper left")
    axes[1,0].set_xlabel("N Measurement")
    axes[1,0].set_ylabel("Abs Current Sum [uA]")
    axes[1,0].grid(True, alpha=0.3)
    combined_legend(axes[1,0])
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

def fit_t1_tau_from_summary(qs_tau, qs_contrast, fitting_dic, maxfev):
    try:
        qs_tau = np.atleast_1d(np.asarray(qs_tau, dtype=float))
        qs_contrast, t1_cutoff_idx = normalize_averaged_decay(qs_contrast, return_cutoff=True)
        qs_tau = qs_tau[t1_cutoff_idx:]
    except (TypeError, ValueError):
        return np.nan
    if len(qs_contrast) < 3 or len(qs_tau) < 3:
        return np.nan
    try:
        min_tau = positive_tau_lower_bound(qs_tau)
        p0 = [qs_tau[-1] / 3, 1.0]
        if fitting_dic.get("use_bounds") == True:
            bounds = ([min_tau, 0.1], [qs_tau[-1] * 10, 3.0])
        else:
            bounds = (-np.inf, np.inf)
        popt, _ = curve_fit(stretched_exp, qs_tau, qs_contrast, p0=p0, bounds=bounds, maxfev=maxfev)
        tau = float(popt[0])
        if tau < 0 or not np.isfinite(tau):
            return np.nan
        return tau
    except Exception:
        return np.nan

def t1_time_by_average_bins(qs_data_reader, averages_per_bin=3000, fitting_dic=None, maxfev=50):
    fitting_dic = fitting_dic or {}
    averages_per_bin = max(int(averages_per_bin), 1)
    t_begin, t_end, signal_rows = qs_data_reader._load_raw_data()
    if len(t_begin) == 0:
        return np.array([], dtype=float), np.array([], dtype=float), []

    rows_per_average = 1
    if qs_data_reader.averaging_mode == "sum":
        rows_per_average = max(len(qs_data_reader._x_values_array), 1)
    rows_per_bin = averages_per_bin * rows_per_average
    time_values = []
    tau_values = []
    averages_by_bin = []
    for start in range(0, len(t_begin), rows_per_bin):
        stop = min(start + rows_per_bin, len(t_begin))
        if stop <= start:
            continue
        row_indices = np.arange(start, stop, dtype=int)
        t_begin_rows = t_begin[row_indices]
        t_end_rows = t_end[row_indices]
        signal_rows_bin = signal_rows[row_indices]
        if qs_data_reader.averaging_mode == "sum":
            summary = qs_data_reader.evaluate_sum_mode(row_indices, t_begin_rows, t_end_rows, signal_rows_bin)
        else:
            summary = qs_data_reader.evaluate_spread_mode(t_begin_rows, t_end_rows, signal_rows_bin)
        tau = fit_t1_tau_from_summary(summary["x_data"], summary["contrast_mean"], fitting_dic, maxfev)
        time_values.append(float(np.nanmean((t_begin_rows + t_end_rows) / 2)))
        tau_values.append(tau)
        averages_by_bin.append((stop - start) / rows_per_average)

    return np.asarray(time_values, dtype=float), np.asarray(tau_values, dtype=float), averages_by_bin

def trim_t1_time_bins(time_values, tau_values, averages_by_bin, time_min, time_max):
    """Keep T1 bins whose timestamps fall within the displayed EC interval."""
    time_values = np.asarray(time_values, dtype=float)
    tau_values = np.asarray(tau_values, dtype=float)
    averages_by_bin = list(averages_by_bin)
    if len(time_values) != len(tau_values) or len(time_values) != len(averages_by_bin):
        raise ValueError("T1 time, fit, and average-bin data must have the same length")

    in_ec_range = (
        np.isfinite(time_values)
        & (time_values >= float(time_min))
        & (time_values <= float(time_max))
    )
    return (
        time_values[in_ec_range],
        tau_values[in_ec_range],
        [count for count, keep in zip(averages_by_bin, in_ec_range) if keep],
    )

def lightlevel_by_time_bins(qs_data_reader, time_edges):
    t_begin, t_end, signal_rows = qs_data_reader._load_raw_data()
    n_bins = max(len(time_edges) - 1, 0)
    if n_bins == 0 or len(t_begin) == 0:
        return np.full(n_bins, np.nan)

    row_reference_mean = np.mean(signal_rows[:, :, 0], axis=1)
    order = np.argsort(t_begin)
    t_begin_sorted = t_begin[order]
    t_end_sorted = t_end[order]
    row_reference_mean = row_reference_mean[order]

    if getattr(qs_data_reader, "averaging_mode", None) == "sum" and len(qs_data_reader._x_values_array) > 0:
        x_values = qs_data_reader._x_values_array[np.arange(len(t_begin)) % len(qs_data_reader._x_values_array)]
        x_values = x_values[order]
    else:
        x_values = None

    lightlevels = np.full(n_bins, np.nan)
    for i, (bin_start, bin_end) in enumerate(zip(time_edges[:-1], time_edges[1:])):
        start_idx = np.searchsorted(t_begin_sorted, bin_start, side="left")
        stop_idx = np.searchsorted(t_begin_sorted, bin_end, side="right")
        if stop_idx <= start_idx:
            continue

        candidate_end = t_end_sorted[start_idx:stop_idx]
        in_bin = candidate_end <= bin_end
        if not np.any(in_bin):
            continue

        refs = row_reference_mean[start_idx:stop_idx][in_bin]
        if x_values is None:
            lightlevels[i] = float(np.mean(refs))
            continue

        xs = x_values[start_idx:stop_idx][in_bin]
        group_means = [float(np.mean(refs[xs == key])) for key in dict.fromkeys(xs.tolist())]
        lightlevels[i] = float(np.mean(group_means)) if group_means else np.nan

    return lightlevels

def contrast_scale_by_time_bins(qs_data_reader, time_edges):
    t_begin, t_end, signal_rows = qs_data_reader._load_raw_data()
    n_bins = max(len(time_edges) - 1, 0)
    if n_bins == 0 or len(t_begin) == 0:
        return np.full(n_bins, np.nan)

    order = np.argsort(t_begin)
    t_begin_sorted = t_begin[order]
    t_end_sorted = t_end[order]
    signal_rows_sorted = signal_rows[order]

    if getattr(qs_data_reader, "averaging_mode", None) == "sum" and len(qs_data_reader._x_values_array) > 0:
        x_values = qs_data_reader._x_values_array[np.arange(len(t_begin)) % len(qs_data_reader._x_values_array)]
        x_values = x_values[order]
    else:
        x_values = None

    contrast_scales = np.full(n_bins, np.nan)
    for i, (bin_start, bin_end) in enumerate(zip(time_edges[:-1], time_edges[1:])):
        start_idx = np.searchsorted(t_begin_sorted, bin_start, side="left")
        stop_idx = np.searchsorted(t_begin_sorted, bin_end, side="right")
        if stop_idx <= start_idx:
            continue

        candidate_end = t_end_sorted[start_idx:stop_idx]
        in_bin = candidate_end <= bin_end
        if not np.any(in_bin):
            continue

        rows = signal_rows_sorted[start_idx:stop_idx][in_bin]
        if x_values is None:
            refs = np.mean(rows[:, :, 0], axis=0)
            vals = np.mean(rows[:, :, 1], axis=0)
        else:
            xs = x_values[start_idx:stop_idx][in_bin]
            refs = []
            vals = []
            for key in dict.fromkeys(xs.tolist()):
                group = rows[xs == key]
                refs.append(float(np.mean(group[:, :, 0])))
                vals.append(float(np.mean(group[:, :, 1])))
            refs = np.asarray(refs, dtype=float)
            vals = np.asarray(vals, dtype=float)

        if len(refs) == 0:
            continue

        raw_contrast = np.divide(
            vals,
            refs,
            out=np.zeros_like(vals, dtype=float),
            where=refs != 0,
        )
        cutoff_idx = int(np.argmin(raw_contrast))
        contrast_window = 1.0 - raw_contrast[cutoff_idx:]
        contrast_scales[i] = float(np.max(contrast_window)) if contrast_window.size else np.nan

    return contrast_scales
