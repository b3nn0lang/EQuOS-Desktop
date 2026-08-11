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
DATA_FOLDER = Path(r"G:\Bucherlab\Surface_Electrochemistry_EPR\Data\NV\2026_08_04_2J_Ru3-1mM-KCl-50mM")

# The shared part after ``ECQS_CV_`` / ``ECQS_T1_``, including an annotation
# if there is one.  Example: "20260730143000_my_sample".
MEASUREMENT_NAME = "20260804134206"


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
# Each entry defines ``[cycle_difference, cycle_step]``: the number of cycles
# included in an evaluation and the number of cycles by which its start is
# advanced.  For example, ``[50, 10]`` evaluates 1-50, 11-60, 21-70, etc.
# until the measurement ends.  The final block can be shorter.  A single
# integer remains supported as a shorthand for non-overlapping blocks, e.g.
# ``50`` is equivalent to ``[50, 50]``.
CYCLE_DIFFERENCES = [[20, 20], [50, 50], [100, 100], [200, 200]]

# Each cycle block is evaluated once for every T1 potential step below.
T1_POTENTIAL_STEPS = [0.02, 0.050, 0.100, 0.2]  # V; 0.010 V = 10 mV

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
    """Evaluate configured cycle windows for every cycle/step setting and T1 step."""
    if not CYCLE_DIFFERENCES or not T1_POTENTIAL_STEPS:
        raise ValueError("CYCLE_DIFFERENCES and T1_POTENTIAL_STEPS must not be empty.")

    cycle_windows = [normalise_cycle_window(configured_window) for configured_window in CYCLE_DIFFERENCES]

    measurement_count = potentiostat.get_measurement_count()
    selections = [
        (cycle_difference, cycle_step, cycle_start, min(cycle_start + cycle_difference - 1, measurement_count), t1_step)
        for cycle_difference, cycle_step in cycle_windows
        for cycle_start in range(1, measurement_count + 1, cycle_step)
        for t1_step in T1_POTENTIAL_STEPS
    ]
    last_image_path = None
    requested_runs = len(selections)
    print(
        f"Starting {requested_runs} evaluations for cycle windows "
        f"{cycle_windows}; measurement has {measurement_count} cycles."
    )

    for cycle_difference, cycle_step, cycle_start, cycle_end, t1_step in tqdm(
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
            cycle_difference,
            cycle_step,
            t1_step,
        )
        print(
            f"Evaluating cycles {cycle_start}-{cycle_end}, "
            f"cycle difference {cycle_difference}, cycle step {cycle_step}, "
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


def normalise_cycle_window(configured_window: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    """Return a validated ``(cycle_difference, cycle_step)`` pair."""
    if isinstance(configured_window, (list, tuple)):
        if len(configured_window) != 2:
            raise ValueError(
                "Each list/tuple in CYCLE_DIFFERENCES must be "
                "[cycle_difference, cycle_step]."
            )
        cycle_difference, cycle_step = configured_window
    else:
        cycle_difference = cycle_step = configured_window

    cycle_difference = int(cycle_difference)
    cycle_step = int(cycle_step)
    if cycle_difference < 1 or cycle_step < 1:
        raise ValueError("Cycle difference and cycle step must both be positive.")
    return cycle_difference, cycle_step


def selection_image_name(
    cycle_start: int,
    cycle_end: int,
    cycle_difference: int,
    cycle_step: int,
    t1_step: float,
) -> str:
    """Return a filename that records every parameter used for one evaluation."""
    t1_step_mv = t1_step * 1000
    t1_step_label = f"{t1_step_mv:g}".replace("-", "neg").replace(".", "p")
    return (
        f"cycles_{cycle_start:03d}-{cycle_end:03d}"
        f"_difference_{cycle_difference:03d}"
        f"_cycle-step_{cycle_step:03d}"
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
