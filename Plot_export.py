"""Export the numerical data stored in Matplotlib figures to a wide CSV file."""

import csv
import os
import re

import numpy as np
from matplotlib.collections import PathCollection, PolyCollection


def _safe_name(value, fallback):
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_")
    return (text[:80] or fallback)


def _artist_label(artist, fallback):
    label = artist.get_label() if hasattr(artist, "get_label") else ""
    if not label or str(label).startswith("_"):
        return fallback
    return _safe_name(label, fallback)


def _flat_values(values):
    values = np.ma.asarray(values).reshape(-1)
    mask = np.ma.getmaskarray(values)
    if np.any(mask):
        unmasked = np.asarray(values.data, dtype=object)
        unmasked[mask] = np.nan
        return unmasked.tolist()
    return np.asarray(values.data).tolist()


def figure_data_columns(fig):
    """Return ordered ``(column_name, values)`` entries for plotted artists."""
    columns = []
    used_names = {}

    def add_column(name, values):
        name = _safe_name(name, "value")
        used_names[name] = used_names.get(name, 0) + 1
        if used_names[name] > 1:
            name = f"{name}_{used_names[name]}"
        columns.append((name, _flat_values(values)))

    def add_xy(prefix, x_values, y_values, x_name, y_name):
        x_values = _flat_values(x_values)
        y_values = _flat_values(y_values)
        length = min(len(x_values), len(y_values))
        if length == 0:
            return
        add_column(f"{prefix}__{x_name}", x_values[:length])
        add_column(f"{prefix}__{y_name}", y_values[:length])

    for axis_number, axis in enumerate(fig.axes, start=1):
        axis_name = _safe_name(
            axis.get_title() or axis.get_ylabel() or axis.get_xlabel(),
            f"axis_{axis_number:02d}",
        )
        axis_prefix = f"axis_{axis_number:02d}_{axis_name}"
        x_name = _safe_name(axis.get_xlabel(), "x")
        y_name = _safe_name(axis.get_ylabel(), "y")

        for line_number, line in enumerate(axis.lines, start=1):
            prefix = f"{axis_prefix}__{_artist_label(line, f'line_{line_number:02d}')}"
            add_xy(prefix, line.get_xdata(), line.get_ydata(), x_name, y_name)

        for collection_number, collection in enumerate(axis.collections, start=1):
            fallback = f"collection_{collection_number:02d}"
            prefix = f"{axis_prefix}__{_artist_label(collection, fallback)}"
            if isinstance(collection, PathCollection):
                offsets = np.ma.asarray(collection.get_offsets())
                if offsets.ndim == 2 and offsets.shape[1] >= 2:
                    add_xy(prefix, offsets[:, 0], offsets[:, 1], x_name, y_name)
            elif isinstance(collection, PolyCollection):
                vertices = []
                for path in collection.get_paths():
                    if vertices:
                        vertices.append([np.nan, np.nan])
                    vertices.extend(path.vertices)
                if vertices:
                    vertices = np.asarray(vertices)
                    add_xy(prefix, vertices[:, 0], vertices[:, 1], x_name, y_name)

        for container_number, container in enumerate(axis.containers, start=1):
            patches = getattr(container, "patches", None)
            if not patches:
                continue
            prefix = f"{axis_prefix}__{_artist_label(container, f'bars_{container_number:02d}')}"
            centers = [patch.get_x() + patch.get_width() / 2 for patch in patches]
            heights = [patch.get_height() for patch in patches]
            add_xy(prefix, centers, heights, x_name, y_name)

        for image_number, plotted_image in enumerate(axis.images, start=1):
            values = np.ma.asarray(plotted_image.get_array()).reshape(-1)
            prefix = f"{axis_prefix}__image_{image_number:02d}"
            add_xy(prefix, np.arange(values.size), values, "pixel_index", "value")

    return columns


def export_figure_data_csv(fig, csv_path):
    """Write all line, scatter, area, bar, and image data from *fig* to CSV."""
    columns = figure_data_columns(fig)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        if not columns:
            writer.writerow(["No plotted data"])
            return 0
        writer.writerow([name for name, _ in columns])
        row_count = max(len(values) for _, values in columns)
        for row_number in range(row_count):
            writer.writerow([
                values[row_number] if row_number < len(values) else ""
                for _, values in columns
            ])
    return len(columns)
