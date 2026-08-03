# -*- coding: utf-8 -*-
"""
Viewer GUI for finished EQuOS measurements.
"""
import gc
import os
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Data_evaluator import ec_data_displayer, ecqs_data_displayer, qs_data_displayer
from Equos_logic import ECQS_cache, parse_file_folder
from Potentiostat import Potentiostat
from QS_data_reader import QS_data_reader


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Configure>", self.on_resize)
        self.data_folder = None
        self.current_measurement_method = None
        self.qs_file_name = ""
        self.ec_file_name = ""
        self.qs_data_reader = None
        self.potentiostat = None
        self.ecqs_cache = ECQS_cache()
        self.fig = None
        self.final_canvas = None
        self.loader = None
        self.fitting_frames = {}
        self.after(0, self.layout)

    def layout(self):
        self.title("EQuOS Viewer")
        self.state("zoomed")
        header = tk.Frame(self)
        header.pack(pady=(8, 2))
        try:
            self.iconbitmap("logo/EQuOS_icon.ico")
            self.photo = tk.PhotoImage(file="logo/EQuOS_logo.png")
            tk.Label(header, image=self.photo).pack(side=tk.LEFT)
        except Exception:
            pass
        tk.Label(header, text="Viewer", font=("Arial", 36, "bold")).pack(side=tk.LEFT, padx=18)

        toolbar = tk.Frame(self)
        toolbar.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(toolbar, text="Change directory", command=self.on_folder_send).pack(side=tk.LEFT)
        self.folder_label = tk.Label(toolbar, text="No folder selected", anchor="w")
        self.folder_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)
        page_canvas = tk.Canvas(body, highlightthickness=0)
        page_scrollbar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=page_canvas.yview)
        page_canvas.configure(yscrollcommand=page_scrollbar.set)
        page_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        page_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.page = tk.Frame(page_canvas)
        self.page_window = page_canvas.create_window((0, 0), window=self.page, anchor="nw")
        self.page.bind("<Configure>", lambda event: page_canvas.configure(scrollregion=page_canvas.bbox("all")))
        page_canvas.bind("<Configure>", lambda event: page_canvas.itemconfigure(self.page_window, width=event.width))
        page_canvas.bind("<MouseWheel>", lambda event: page_canvas.yview_scroll(-1 * (event.delta // 120), "units"))

        controls = tk.Frame(self.page)
        controls.pack(fill=tk.X, padx=10, pady=8)

        self.load = tk.Frame(controls)
        self.load.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        loader_title = tk.Label(self.load, text="Finished measurements", font=("Arial", 12, "bold"))
        loader_title.pack(anchor="w")
        self.loader = ttk.Treeview(
            self.load,
            columns=("type", "method", "date", "time", "annotation"),
            show="headings",
            height=8,
        )
        for column, width in (
            ("type", 60),
            ("method", 80),
            ("date", 100),
            ("time", 90),
            ("annotation", 180),
        ):
            self.loader.heading(column, text=column.title())
            self.loader.column(column, width=width, stretch=column == "annotation")
        self.loader.bind("<Double-1>", self.file_selected)
        self.loader.pack(fill=tk.BOTH, expand=True, pady=8)

        self.fitting_params_frame = tk.Frame(controls, bd=2, relief="groove")
        self.fitting_params_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        tk.Label(self.fitting_params_frame, text="Data evaluation", font=("Arial", 12, "bold")).pack()
        self.layout_fitting_frames()
        tk.Button(self.fitting_params_frame, text="Save Figure", command=self.save_figure).pack(pady=6)
        self.eval_info_text = tk.Text(self.fitting_params_frame, height=8)
        self.eval_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=6)

        plot_section = tk.Frame(self.page)
        plot_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 12))
        tk.Label(plot_section, text="Plot", font=("Arial", 12, "bold")).pack(anchor="w")
        self.final_plot_container = tk.Frame(plot_section, height=720)
        self.final_plot_container.pack(fill=tk.BOTH, expand=True)
        self.final_plot_container.pack_propagate(False)

    def layout_fitting_frames(self):
        esr_frame = self.add_fitting_frame("ESR", "ESR peak finding")
        self.esr_fitting_entries = self.layout_fitting_params(
            esr_frame,
            ["Fitting:", "Prominence:", "Distance:"],
            [True, 0.01, 5],
        )
        tk.Button(esr_frame, text="Fit Data", command=self.esr_fitting).pack()

        t1_frame = self.add_fitting_frame("T1", "T1 exp. curve fitting")
        self.t1_fitting_entries = self.layout_fitting_params(
            t1_frame,
            [
                "Fitting:",
                "Use stretched exponential fit:",
                "Stretched exp. fit - Max. tries:",
                "Stretched exp. fit - use Bounds:",
                "Show stretched exp. fit:",
                "Use Biexpontential fit:",
                "biexp. fit - Max. tries:",
                "biexp.fit - use Bounds:",
                "Show biexp. fit:",
            ],
            [True, True, 20000, True, True, False, 20000, True, True],
        )
        tk.Button(t1_frame, text="Fit Data", command=self.t1_fitting).pack()

        rabi_frame = self.add_fitting_frame("Rabi", "Rabi sinus curve fitting")
        self.rabi_fitting_entries = self.layout_fitting_params(
            rabi_frame,
            ["Fitting:", "Use Sinus_fit:", "Show Sinus fit:", "Use Bounds:", "Max. Tries:", "Use Peakfinding:", "Prominence:", "Distance:"],
            [True, True, True, True, 50000, True, 0.03, 10],
        )
        tk.Button(rabi_frame, text="Fit Data", command=self.rabi_fitting).pack()

        self.fitting_frames["LaserOff"] = self.add_fitting_frame("LaserOff", "Laser Off data")
        self.fitting_frames["LaserOn"] = self.add_fitting_frame("LaserOn", "Laser On data")

        for method in ("CV", "SWV", "LSV", "CA"):
            frame = self.add_fitting_frame(method, f"{method} peak finding")
            if method == "CA":
                labels = ["Fitting:", "From time [s]:", "To time [s] (0 = last):"]
                defaults = [False, 0, 0]
            else:
                labels = ["Fitting:", "Prominence:", "Distance:"]
                defaults = [True, 1e-6, 10]
            entries = self.layout_fitting_params(frame, labels, defaults)
            setattr(self, f"{method.lower()}_fitting_entries", entries)
            tk.Button(frame, text="Fit Data", command=getattr(self, f"{method.lower()}_fitting")).pack()

        ecqs_labels = ["Fitting:", "Prominence:", "Distance:", "Cutoff Rate", "Use Bounds:", "Max. Tries:", "Use R^2 Threshold:", "Min. R^2:", "From measurement N:", "To measurement N (0 = last):", "T1 Potential Step [mV]:"]
        ecqs_defaults = [True, 1e-6, 10, 0.5, False, 5000, False, 0.95, 1, 0, 50]
        for method in ("CV-T1", "SWV-T1", "LSV-T1"):
            frame = self.add_fitting_frame(method, f"{method} fitting")
            attr = method.lower().replace("-", "_") + "_fitting_entries"
            setattr(self, attr, self.layout_fitting_params(frame, ecqs_labels, ecqs_defaults))
            tk.Button(frame, text="Fit Data", command=getattr(self, method.lower().replace("-", "_") + "_fitting")).pack()

        ca_t1_labels = ["Fitting:", "Cutoff Rate", "Use Bounds:", "Max. Tries:", "Use R^2 Threshold:", "Min. R^2:", "T1 averages/datapoint:", "Time bin size [s]:", "Save individual T1 plots:", "From time [s]:", "To time [s] (0 = last):"]
        ca_t1_defaults = [True, 0.5, False, 5000, False, 0.95, 300, 0, True, 0, 0]
        ca_t1_frame = self.add_fitting_frame("CA-T1", "CA-T1 fitting")
        self.ca_t1_fitting_entries = self.layout_fitting_params(ca_t1_frame, ca_t1_labels, ca_t1_defaults)
        tk.Button(ca_t1_frame, text="Fit Data", command=self.ca_t1_fitting).pack()

    def add_fitting_frame(self, key: str, title: str) -> tk.Frame:
        frame = tk.Frame(self.fitting_params_frame)
        self.fitting_frames[key] = frame
        tk.Label(frame, text=title).pack()
        return frame

    def layout_fitting_params(self, frame, parameter_list: list, default_list: list) -> list:
        param_frame = tk.Frame(frame)
        param_frame.pack()
        entries = []
        for row, (label_text, default) in enumerate(zip(parameter_list, default_list)):
            tk.Label(param_frame, text=label_text).grid(row=row, column=0, padx=5, pady=4, sticky="w")
            if isinstance(default, bool):
                var = tk.BooleanVar(value=default)
                tk.Checkbutton(param_frame, variable=var).grid(row=row, column=1, padx=5, pady=4)
                entries.append(("bool", var))
            else:
                entry = tk.Entry(param_frame, width=12)
                entry.insert(0, default)
                entry.grid(row=row, column=1, padx=5, pady=4)
                entries.append(("entry", entry))
        return entries

    def get_fitting_value(self, item):
        typ, obj = item
        value = obj.get()
        if typ == "entry":
            try:
                return float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                return value
        return value

    def on_folder_send(self):
        selected = filedialog.askdirectory()
        if not selected or not os.path.isdir(selected):
            return
        self.data_folder = selected
        self.folder_label.config(text=selected)
        self.update_file_loader()

    def update_file_loader(self):
        if not self.data_folder:
            return
        self.loader.delete(*self.loader.get_children())
        measurements = parse_file_folder(self.data_folder)
        measurements.sort(key=lambda x: (x["date"], x["time"]), reverse=True)
        for measurement in measurements:
            self.loader.insert(
                "",
                "end",
                iid=measurement["file_name"],
                values=(
                    measurement["type"],
                    measurement["method"],
                    measurement["date"],
                    measurement["time"],
                    measurement["annotation"],
                ),
            )

    def file_selected(self, event):
        item_id = self.loader.selection()
        if not item_id:
            return
        iid = item_id[0]
        values = self.loader.item(iid, "values")
        self.cleanup_measurement_state()
        if values[0] == "QS":
            self.load_qs_measurement(iid, values[1])
        elif values[0] == "EC":
            self.load_ec_measurement(iid, values[1])
        elif values[0] == "ECQS":
            self.load_ecqs_measurement(values)

    def load_qs_measurement(self, file_name: str, method: str):
        self.qs_file_name = file_name
        self.current_measurement_method = method
        self.qs_data_reader = QS_data_reader(self.data_folder, file_name)
        self.qs_data_reader.start_data_building()
        self.eval_qs_figure()

    def load_ec_measurement(self, file_name: str, method: str):
        self.ec_file_name = file_name
        self.current_measurement_method = method
        self.potentiostat = Potentiostat(self.data_folder)
        self.potentiostat.load_measurement_from_file(file_name)
        self.eval_ec_figure()

    def load_ecqs_measurement(self, values):
        ecqs_files = []
        used_methods = []
        for all_item_id in self.loader.get_children():
            all_values = self.loader.item(all_item_id, "values")
            if all_values[2] == values[2] and all_values[3] == values[3]:
                ecqs_files.append(all_item_id)
                used_methods.append(all_values[1])
        if "CV" in used_methods and "T1" in used_methods:
            self.current_measurement_method = "CV-T1"
        elif "SWV" in used_methods and "T1" in used_methods:
            self.current_measurement_method = "SWV-T1"
        elif "LSV" in used_methods and "T1" in used_methods:
            self.current_measurement_method = "LSV-T1"
        elif "CA" in used_methods and "T1" in used_methods:
            self.current_measurement_method = "CA-T1"
        else:
            return
        for file_name in ecqs_files:
            method = file_name.split("_")[1]
            if method in ("CV", "SWV", "LSV", "CA"):
                self.ec_file_name = file_name
            elif method == "T1":
                self.qs_file_name = file_name
        self.qs_data_reader = QS_data_reader(self.data_folder, self.qs_file_name)
        self.qs_data_reader.start_data_building()
        self.potentiostat = Potentiostat(self.data_folder)
        self.potentiostat.load_measurement_from_file(self.ec_file_name)
        self.eval_ecqs_figure()

    def cleanup_measurement_state(self):
        if self.qs_data_reader is not None:
            self.qs_data_reader.release_resources()
            self.qs_data_reader = None
        if self.potentiostat is not None:
            self.potentiostat.release_resources()
            self.potentiostat = None
        self.ecqs_cache.clear()
        self.reset_final_canvas()
        self.fig = None
        gc.collect()

    def reset_final_canvas(self):
        if self.final_canvas is not None:
            old_fig = self.final_canvas.figure
            self.final_canvas.get_tk_widget().destroy()
            self.final_canvas = None
            if old_fig is not None:
                plt.close(old_fig)

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

    def fitting_parameter_update(self):
        for old_fit in self.fitting_frames.values():
            old_fit.pack_forget()
        frame = self.fitting_frames.get(self.current_measurement_method)
        if frame is not None:
            frame.pack(padx=10, pady=10, fill="both", expand=False)

    def measurement_info_update(self, text: dict = None) -> None:
        text = text or {}
        self.eval_info_text.delete("1.0", tk.END)
        for key, val in text.items():
            self.eval_info_text.insert(tk.END, f"{key}: {val}\n")

    def get_evaluation_plot_folder(self) -> str | None:
        if not self.data_folder:
            return None
        if self.current_measurement_method in ("CV", "SWV", "LSV", "CA"):
            measurement_name = self.ec_file_name
        elif self.current_measurement_method in ("CV-T1", "SWV-T1", "LSV-T1", "CA-T1"):
            measurement_name = self.qs_file_name or self.ec_file_name
        else:
            measurement_name = self.qs_file_name
        return os.path.join(self.data_folder, measurement_name) if measurement_name else None

    def safe_plot_file_part(self, text: str) -> str:
        cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(text))
        return cleaned.strip("_") or "axis"

    def save_figure_without_titles(self, file_path: str) -> None:
        suptitle = getattr(self.fig, "_suptitle", None)
        suptitle_text = suptitle.get_text() if suptitle is not None else None
        axis_titles = [axis.get_title() for axis in self.fig.axes]
        try:
            if suptitle is not None:
                suptitle.set_text("")
            for axis in self.fig.axes:
                axis.set_title("")
            self.fig.savefig(file_path, dpi=300, bbox_inches="tight")
        finally:
            if suptitle is not None:
                suptitle.set_text(suptitle_text)
            for axis, title in zip(self.fig.axes, axis_titles):
                axis.set_title(title)

    def save_axis_as_standalone_figure(self, source_axis, file_path: str) -> None:
        axis_fig = Figure(figsize=(7, 5), constrained_layout=True)
        target_axis = axis_fig.subplots()
        for line in source_axis.lines:
            label = line.get_label()
            target_axis.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
                markerfacecolor=line.get_markerfacecolor(),
                markeredgecolor=line.get_markeredgecolor(),
                alpha=line.get_alpha(),
                label=label if label and not label.startswith("_") else None,
            )
        for collection in source_axis.collections:
            label = collection.get_label()
            label = label if label and not label.startswith("_") else None
            if isinstance(collection, mcollections.PathCollection):
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    target_axis.scatter(
                        offsets[:, 0],
                        offsets[:, 1],
                        s=collection.get_sizes(),
                        c=collection.get_facecolors(),
                        edgecolors=collection.get_edgecolors(),
                        alpha=collection.get_alpha(),
                        label=label,
                    )
            elif isinstance(collection, mcollections.PolyCollection):
                copied_collection = mcollections.PolyCollection(
                    [path.vertices for path in collection.get_paths()],
                    facecolors=collection.get_facecolors(),
                    edgecolors=collection.get_edgecolors(),
                    linewidths=collection.get_linewidths(),
                    linestyles=collection.get_linestyles(),
                    alpha=collection.get_alpha(),
                    label=label,
                )
                target_axis.add_collection(copied_collection)
        for patch in source_axis.patches:
            if isinstance(patch, mpatches.Rectangle) and (patch.get_width() != 0 or patch.get_height() != 0):
                copied_patch = mpatches.Rectangle(
                    patch.get_xy(),
                    patch.get_width(),
                    patch.get_height(),
                    angle=patch.angle,
                    facecolor=patch.get_facecolor(),
                    edgecolor=patch.get_edgecolor(),
                    linewidth=patch.get_linewidth(),
                    alpha=patch.get_alpha(),
                    label=patch.get_label() if patch.get_label() and not patch.get_label().startswith("_") else None,
                )
                target_axis.add_patch(copied_patch)
        for image in source_axis.images:
            target_axis.imshow(
                image.get_array(),
                extent=image.get_extent(),
                origin=image.origin,
                cmap=image.get_cmap(),
                norm=image.norm,
                alpha=image.get_alpha(),
                aspect=source_axis.get_aspect(),
            )
        target_axis.set_xlabel(source_axis.get_xlabel())
        target_axis.set_ylabel(source_axis.get_ylabel())
        target_axis.set_xscale(source_axis.get_xscale())
        target_axis.set_yscale(source_axis.get_yscale())
        target_axis.set_xlim(source_axis.get_xlim())
        target_axis.set_ylim(source_axis.get_ylim())
        if any(line.get_visible() for line in source_axis.get_xgridlines() + source_axis.get_ygridlines()):
            target_axis.grid(True, alpha=0.3)
        handles, labels = target_axis.get_legend_handles_labels()
        if handles and labels:
            target_axis.legend()
        axis_fig.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(axis_fig)

    def save_evaluation_plots(self) -> None:
        if self.fig is None:
            return
        plot_folder = self.get_evaluation_plot_folder()
        if plot_folder is None:
            return
        try:
            os.makedirs(plot_folder, exist_ok=True)
            self.save_figure_without_titles(os.path.join(plot_folder, "final_figure.png"))
            for index, axis in enumerate(self.fig.axes, start=1):
                if not axis.get_visible():
                    continue
                title = axis.get_title() or axis.get_ylabel() or axis.get_xlabel() or f"axis_{index}"
                file_name = f"axis_{index:02d}_{self.safe_plot_file_part(title)}.png"
                self.save_axis_as_standalone_figure(axis, os.path.join(plot_folder, file_name))
        except Exception as exc:
            print(f"Automatic plot export failed: {exc}")

    def eval_qs_figure(self, fitting_params: dict = None):
        fitting_params = fitting_params or {"fitting": self.current_measurement_method not in ("LaserOff", "LaserOn")}
        self.fig, _, _, _, info = qs_data_displayer(self.qs_data_reader, fitting_params)
        self._replace_final_canvas()
        self.fitting_parameter_update()
        self.measurement_info_update(info)
        self.save_evaluation_plots()

    def eval_ec_figure(self, fitting_params: dict = None):
        self.fig, _, _, _, info = ec_data_displayer(self.potentiostat, fitting_params or {"fitting": True})
        self._replace_final_canvas()
        self.fitting_parameter_update()
        self.measurement_info_update(info)
        self.save_evaluation_plots()

    def eval_ecqs_figure(self, fitting_params: dict = None):
        self.ecqs_cache = ECQS_cache()
        self.fig, _, _, _, info, _ = ecqs_data_displayer(
            self.qs_data_reader,
            self.potentiostat,
            fitting_params or {"fitting": True},
            self.ecqs_cache,
        )
        self._replace_final_canvas()
        self.fitting_parameter_update()
        self.measurement_info_update(info)
        self.save_evaluation_plots()

    def save_figure(self):
        if self.fig is None:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF file", "*.pdf"), ("SVG file", "*.svg"), ("All files", "*.*")],
            title="Save figure as...",
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches="tight")
            print("Saved to:", file_path)

    def esr_fitting(self):
        values = [self.get_fitting_value(item) for item in self.esr_fitting_entries]
        self.eval_qs_figure({"fitting": values[0], "esr_prominence": values[1], "esr_distance": values[2]})

    def rabi_fitting(self):
        values = [self.get_fitting_value(item) for item in self.rabi_fitting_entries]
        self.eval_qs_figure(
            {
                "fitting": values[0],
                "rabi_sin_fit": values[1],
                "rabi_show_fit": values[2],
                "rabi_use_bounds": values[3],
                "rabi_max_tries": values[4],
                "rabi_find_peak": values[5],
                "rabi_prominence": values[6],
                "rabi_distance": values[7],
            }
        )

    def t1_fitting(self):
        values = [self.get_fitting_value(item) for item in self.t1_fitting_entries]
        self.eval_qs_figure(
            {
                "fitting": values[0],
                "t1_stretched_exp_fit": values[1],
                "t1_max_stretch_tries": values[2],
                "t1_use_stretch_bounds": values[3],
                "t1_show_stretch_fit": values[4],
                "t1_bi_exp_fit": values[5],
                "t1_max_bi_tries": values[6],
                "t1_use_bi_bounds": values[7],
                "t1_show_bi_fit": values[8],
            }
        )

    def cv_fitting(self):
        values = [self.get_fitting_value(item) for item in self.cv_fitting_entries]
        self.eval_ec_figure({"fitting": values[0], "prominence": values[1], "distance": values[2]})

    def swv_fitting(self):
        values = [self.get_fitting_value(item) for item in self.swv_fitting_entries]
        self.eval_ec_figure({"fitting": values[0], "prominence": values[1], "distance": values[2]})

    def lsv_fitting(self):
        values = [self.get_fitting_value(item) for item in self.lsv_fitting_entries]
        self.eval_ec_figure({"fitting": values[0], "prominence": values[1], "distance": values[2]})

    def ca_fitting(self):
        values = [self.get_fitting_value(item) for item in self.ca_fitting_entries]
        self.eval_ec_figure({"fitting": values[0], "time_from": values[1], "time_to": values[2]})

    def ecqs_fitting_dict(self, entries):
        values = [self.get_fitting_value(item) for item in entries]
        return {
            "ec_fitting": values[0],
            "cutoff_rate": values[3],
            "max_tries": values[5],
            "use_bounds": values[4],
            "prominence": values[1],
            "distance": values[2],
            "use_r_square_threshold": values[6],
            "r_square_threshold": values[7],
            "measurement_from": values[8],
            "measurement_to": values[9],
            "t1_potential_step": values[10] / 1000,
        }

    def cv_t1_fitting(self):
        self.eval_ecqs_figure(self.ecqs_fitting_dict(self.cv_t1_fitting_entries))

    def swv_t1_fitting(self):
        self.eval_ecqs_figure(self.ecqs_fitting_dict(self.swv_t1_fitting_entries))

    def lsv_t1_fitting(self):
        self.eval_ecqs_figure(self.ecqs_fitting_dict(self.lsv_t1_fitting_entries))

    def ca_t1_fitting(self):
        values = [self.get_fitting_value(item) for item in self.ca_t1_fitting_entries]
        self.eval_ecqs_figure(
            {
                "fitting": values[0],
                "cutoff_rate": values[1],
                "use_bounds": values[2],
                "max_tries": values[3],
                "use_r_square_threshold": values[4],
                "r_square_threshold": values[5],
                "t1_averages_per_datapoint": values[6],
                "time_bin_size": values[7],
                "save_detail_plots": values[8],
                "time_from": values[9],
                "time_to": values[10],
            }
        )

    def on_resize(self, event):
        pass

    def on_close(self):
        self.cleanup_measurement_state()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
