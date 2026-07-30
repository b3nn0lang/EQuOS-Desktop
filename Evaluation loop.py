"""Evaluate one saved CV-T1 measurement outside the EQuOS Viewer.

Set ``DATA_FOLDER`` and ``MEASUREMENT_NAME`` below, then run this file in the
PyCharm console.  A CV-T1 measurement consists of these two files:

    ECQS_CV_<MEASUREMENT_NAME>.csv
    ECQS_T1_<MEASUREMENT_NAME>.yaml  (and its corresponding *_temp.dat file)
"""

from pathlib import Path
import subprocess
import sys
from itertools import product

from matplotlib.figure import Figure

from Data_evaluator import ecqs_data_displayer
from Equos_logic import ECQS_cache
from Potentiostat import Potentiostat
from QS_data_reader import QS_data_reader
from tqdm import tqdm

# --- Measurement to evaluate -------------------------------------------------
# The folder that contains the saved ECQS_CV_* and ECQS_T1_* files.
DATA_FOLDER = Path(r"/Volumes/tus3/001/Bucherlab/Surface_Electrochemistry_EPR/Data/NV/2026_07_29_K3K4FeCN6-250uM-1v1-KCl5mM")

# The shared part after ``ECQS_CV_`` / ``ECQS_T1_``, including an annotation
# if there is one.  Example: "20260730143000_my_sample".
MEASUREMENT_NAME = "20260729115303"


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
# Every combination of these three lists is evaluated.  A cycle difference is
# added to a selected start cycle: e.g. start 10 and difference 5 -> cycles
# 10 through 15 (inclusive).  The script does *not* iterate all recorded
# cycles automatically.
CYCLE_STARTS = [1, 10]
CYCLE_DIFFERENCES = [4, 9]
T1_POTENTIAL_STEPS = [0.010, 0.050, 0.100]  # V; 0.010 V = 10 mV

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
    """Evaluate exactly the configured starts, cycle differences, and T1 steps."""
    if not CYCLE_STARTS or not CYCLE_DIFFERENCES or not T1_POTENTIAL_STEPS:
        raise ValueError("CYCLE_STARTS, CYCLE_DIFFERENCES, and T1_POTENTIAL_STEPS must not be empty.")

    measurement_count = potentiostat.get_measurement_count()
    last_image_path = None
    requested_runs = len(CYCLE_STARTS) * len(CYCLE_DIFFERENCES) * len(T1_POTENTIAL_STEPS)
    print(f"Starting {requested_runs} configured evaluations; measurement has {measurement_count} cycles.")

    selections = product(CYCLE_STARTS, CYCLE_DIFFERENCES, T1_POTENTIAL_STEPS)
    for cycle_start, cycle_difference, t1_step in tqdm(
        selections,
        total=requested_runs,
        desc="Evaluating CV-T1 selections",
        unit="image",
    ):
        cycle_start = int(cycle_start)
        cycle_difference = int(cycle_difference)
        t1_step = float(t1_step)
        cycle_end = cycle_start + cycle_difference

        if cycle_start < 1 or cycle_difference < 0 or t1_step <= 0:
            print(f"Skipping invalid selection: start={cycle_start}, difference={cycle_difference}, step={t1_step} V")
            continue
        if cycle_end > measurement_count:
            print(f"Skipping cycles {cycle_start}-{cycle_end}: measurement has only {measurement_count} cycles.")
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
            cycle_difference,
            t1_step,
        )
        print(
            f"Evaluating cycles {cycle_start}-{cycle_end}, "
            f"difference {cycle_difference}, T1 step {t1_step * 1000:g} mV"
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
        print(f"Saved: {image_path.name}")
        for key, value in info.items():
            print(f"  {key}: {value}")
        last_image_path = image_path

    return last_image_path


def selection_image_name(
    cycle_start: int,
    cycle_end: int,
    cycle_difference: int,
    t1_step: float,
) -> str:
    """Return a filename that records every parameter used for one evaluation."""
    t1_step_mv = t1_step * 1000
    t1_step_label = f"{t1_step_mv:g}".replace("-", "neg").replace(".", "p")
    return (
        f"cycles_{cycle_start:03d}-{cycle_end:03d}"
        f"_difference_{cycle_difference:03d}"
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
