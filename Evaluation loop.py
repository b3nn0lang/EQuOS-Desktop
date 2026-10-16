"""Evaluate one saved CV-T1 measurement outside the EQuOS Viewer.

Set ``DATA_FOLDER`` and ``MEASUREMENT_NAME`` below, then run this file in the
PyCharm console.  A CV-T1 measurement consists of these two files:

    ECQS_CV_<MEASUREMENT_NAME>.csv
    ECQS_T1_<MEASUREMENT_NAME>.yaml  (and its corresponding *_temp.dat file)
"""

from pathlib import Path
import subprocess
import sys

from matplotlib.figure import Figure

from Data_evaluator import ecqs_data_displayer
from Plot_export import export_figure_data_csv
from Equos_logic import ECQS_cache
from Potentiostat import Potentiostat
from QS_data_reader import QS_data_reader
from tqdm import tqdm

# --- Measurement to evaluate -------------------------------------------------
# The folder that contains the saved ECQS_CV_* and ECQS_T1_* files.
DATA_FOLDER = Path(r"G:\Bucherlab\Surface_Electrochemistry_EPR\Data\NV\2026_10_12_K3K4_50mM_KCl_250mM_KOH_1mM_CV")

# The shared part after ``ECQS_CV_`` / ``ECQS_T1_``, including an annotation
# if there is one.  Example: "20260730143000_my_sample".
MEASUREMENT_NAME = "20261012125406_measurement"


# --- Hardcoded fit settings --------------------------------------------------
# These are the same settings as the CV-T1 controls in the viewer.  Adjust
# them here for repeated offline evaluation of measurements.
FITTING_DIC = {
    "ec_fitting": False,
    "cutoff_rate": 0.5,
    "max_tries": 500,
    "use_bounds": False,
    "prominence": 1e-6,
    "distance": 10,
    "use_r_square_threshold": False,
    "r_square_threshold": 0.95,
}


# --- Evaluation sweep --------------------------------------------------------
# Configure each sweep as either:
#   50       -> 1-50, 51-100, 101-150, ...
#   [50, 10] -> 1-50, 11-60, 21-70, ...
# The first value is the evaluation-window size and the optional second value
# is how many cycles to advance between windows. The final window can be shorter.
CYCLE_DIFFERENCES = [[50,10],[100, 10]]

# Each cycle block is evaluated once for every T1 potential step below.
T1_POTENTIAL_STEPS = [0.025, 0.05, 0.1,0.2]  # V; 0.010 V = 10 mV

# Opening every generated figure is impractical for a sweep.  When enabled,
# only the last successfully generated image is opened in Preview.
OPEN_LAST_IMAGE = True


def main() -> None:
    """Evaluate all explicitly configured CV-T1 cycle/step combinations."""
    data_folder = DATA_FOLDER.expanduser().resolve()
    cv_file_name = f"ECQS_CV_{MEASUREMENT_NAME}"
    t1_file_name = f"ECQS_T1_{MEASUREMENT_NAME}"

    if not data_folder.is_dir():
        raise NotADirectoryError(f"Measurement folder does not exist: {data_folder}")
    if not (data_folder / f"{cv_file_name}.csv").is_file():
        raise FileNotFoundError(f"CV data file not found: {data_folder / f'{cv_file_name}.csv'}")
    if not (data_folder / f"{t1_file_name}.yaml").is_file():
        raise FileNotFoundError(f"T1 parameter file not found: {data_folder / f'{t1_file_name}.yaml'}")

    qs_data_reader = QS_data_reader(str(data_folder), t1_file_name)
    potentiostat = Potentiostat(str(data_folder))

    try:
        # Build the T1 data summary before passing it to the joint evaluator.
        qs_data_reader.start_data_building()
        potentiostat.load_measurement_from_file(cv_file_name)

        evaluation_folder = data_folder / f"ec-qs evaluation_{MEASUREMENT_NAME}"
        evaluation_folder.mkdir(exist_ok=True)
        last_image_path = run_evaluation_sweep(
            qs_data_reader,
            potentiostat,
            evaluation_folder,
        )
        if OPEN_LAST_IMAGE and last_image_path is not None:
            open_image_in_system_viewer(last_image_path)
    finally:
        qs_data_reader.release_resources()
        potentiostat.release_resources()


def run_evaluation_sweep(
    qs_data_reader: QS_data_reader,
    potentiostat: Potentiostat,
    evaluation_folder: Path,
) -> Path | None:
    """Evaluate consecutive blocks for every configured difference and T1 step."""
    if not CYCLE_DIFFERENCES or not T1_POTENTIAL_STEPS:
        raise ValueError("CYCLE_DIFFERENCES and T1_POTENTIAL_STEPS must not be empty.")

    cycle_sweeps = [normalize_cycle_sweep(value) for value in CYCLE_DIFFERENCES]

    measurement_count = potentiostat.get_measurement_count()
    selections = [
        (window_size, cycle_advance, cycle_start,
         min(cycle_start + window_size - 1, measurement_count), t1_step)
        for window_size, cycle_advance in cycle_sweeps
        for cycle_start in range(1, measurement_count + 1, cycle_advance)
        for t1_step in T1_POTENTIAL_STEPS
    ]
    last_image_path = None
    requested_runs = len(selections)
    print(
        f"Starting {requested_runs} evaluations for cycle differences "
        f"{CYCLE_DIFFERENCES}; measurement has {measurement_count} cycles."
    )

    for window_size, cycle_advance, cycle_start, cycle_end, t1_step in tqdm(
        selections,
        total=requested_runs,
        desc="Evaluating CV-T1 selections",
        unit="image",
    ):
        t1_step = float(t1_step)

        if t1_step <= 0:
            print(f"Skipping invalid T1 potential step: {t1_step} V")
            continue

        fitting_dic = FITTING_DIC | {
            "measurement_from": cycle_start,
            "measurement_to": cycle_end,
            "t1_potential_step": t1_step,
        }
        # The evaluator treats the mere presence of ``ec_fitting`` as a
        # request for many auxiliary detail images.  Omit it when disabled so
        # this sweep creates only the traceable PNG defined below.
        if not fitting_dic.get("ec_fitting", False):
            fitting_dic.pop("ec_fitting", None)
        image_path = evaluation_folder / selection_image_name(
            cycle_start,
            cycle_end,
            window_size,
            cycle_advance,
            t1_step,
        )
        print(
            f"Evaluating cycles {cycle_start}-{cycle_end}, "
            f"window {window_size}, advance {cycle_advance}, "
            f"T1 step {t1_step * 1000:g} mV"
        )
        output_figure = Figure(figsize=(12, 8), constrained_layout=True)
        fig, _, _, _, info, _ = ecqs_data_displayer(
            qs_data_reader,
            potentiostat,
            fitting_dic,
            ECQS_cache(),
            fig=output_figure,
        )
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        export_figure_data_csv(fig, image_path.with_suffix(".csv"))
        print(f"Saved: {image_path.name}")
        for key, value in info.items():
            print(f"  {key}: {value}")
        last_image_path = image_path

    return last_image_path


def normalize_cycle_sweep(configured_sweep: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    """Return ``(window size, cycle advance)`` for one sweep setting."""
    if isinstance(configured_sweep, (list, tuple)):
        if len(configured_sweep) != 2:
            raise ValueError(
                "A cycle sweep list must contain [window_size, cycle_advance]."
            )
        window_size, cycle_advance = map(int, configured_sweep)
    else:
        window_size = cycle_advance = int(configured_sweep)

    if window_size < 1 or cycle_advance < 1:
        raise ValueError("Cycle window sizes and advances must be positive.")
    return window_size, cycle_advance


def selection_image_name(
    cycle_start: int,
    cycle_end: int,
    window_size: int,
    cycle_advance: int,
    t1_step: float,
) -> str:
    """Return a filename that records every parameter used for one evaluation."""
    t1_step_mv = t1_step * 1000
    t1_step_label = f"{t1_step_mv:g}".replace("-", "neg").replace(".", "p")
    return (
        f"cycles_{cycle_start:03d}-{cycle_end:03d}"
        f"_window_{window_size:03d}"
        f"_advance_{cycle_advance:03d}"
        f"_t1-step_{t1_step_label}mV.png"
    )


def open_image_in_system_viewer(image_path: Path) -> None:
    """Open a saved result in the OS default image viewer."""
    if sys.platform == "darwin":
        subprocess.run(["open", str(image_path)], check=True)
    elif sys.platform == "win32":
        import os
        os.startfile(image_path)  # opens with the default Windows image viewer
    else:
        print("Open the saved PNG with your system image viewer.")


if __name__ == "__main__":
    main()
