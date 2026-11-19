import csv
import math
import json
from pathlib import Path

import numpy as np


def save_signal_csv(path, t, original, filtered):
    if not len(t) == len(original) == len(filtered):
        raise ValueError("Time, original, and filtered arrays must have the same number of samples.")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time (s)", "Original", "Filtered"])
        for row in zip(t, original, filtered):
            writer.writerow(row)


def save_plot(path, figure):
    figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())


def load_signal_csv(path, fs=None):
    csv_path = Path(path)
    if not csv_path.exists():
        raise ValueError(f"CSV file does not exist: {csv_path}")
    try:
        rows = []
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.reader(handle):
                if not row or all(not cell.strip() for cell in row):
                    continue
                rows.append(row)
    except (OSError, csv.Error) as err:
        raise ValueError(f"Could not read CSV file: {err}") from err
    if not rows:
        raise ValueError("CSV file is empty.")

    def parse_row(row):
        values = []
        for cell in row:
            try:
                value = float(cell.strip())
            except (TypeError, ValueError) as err:
                raise ValueError("Text in a number column is invalid.") from err
            if not math.isfinite(value):
                raise ValueError("CSV values must not contain NaN or infinity.")
            values.append(value)
        return values

    first_numeric = []
    try:
        first_numeric = parse_row(rows[0])
        data_rows = rows
    except ValueError:
        data_rows = rows[1:]
        if not data_rows:
            raise

    widths = {len(row) for row in data_rows}
    if len(widths) != 1:
        raise ValueError("CSV rows must have a consistent number of columns.")
    width = widths.pop()
    if width not in (1, 2):
        raise ValueError("CSV must contain one signal column or two time/signal columns.")

    numeric_rows = [parse_row(row) for row in data_rows]
    if len(numeric_rows) < 2:
        raise ValueError("CSV must contain at least 2 samples.")
    if len(numeric_rows) > 1_000_000:
        raise ValueError("CSV contains more than 1,000,000 samples.")

    array = np.asarray(numeric_rows, dtype=float)
    if width == 2:
        time = array[:, 0]
        signal = array[:, 1]
        if np.any(np.diff(time) <= 0):
            raise ValueError("Time values must be strictly increasing.")
        differences = np.diff(time)
        median = float(np.median(differences))
        if median <= 0 or np.max(np.abs(differences - median)) > 1e-6 * max(1.0, median):
            raise ValueError("CSV sampling must be uniform.")
        fs = 1.0 / median
    else:
        signal = array[:, 0]
        if fs is None:
            raise ValueError("Sampling frequency is required for a one-column CSV.")
        try:
            fs = float(fs)
        except (TypeError, ValueError) as err:
            raise ValueError("Sampling frequency must be a positive finite number.") from err
        if not math.isfinite(fs) or fs <= 0:
            raise ValueError("Sampling frequency must be a positive finite number.")
        time = np.arange(signal.size, dtype=float) / fs
    return time, signal, float(fs)


def save_session(path, settings):
    if not isinstance(settings, dict):
        raise ValueError("Session settings must be a dictionary.")
    try:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump({"version": 1, "settings": settings}, handle, indent=2)
    except (OSError, TypeError, ValueError) as err:
        raise ValueError(f"Could not save session: {err}") from err


def load_session(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError as err:
        raise ValueError(f"Session file does not exist: {path}") from err
    except json.JSONDecodeError as err:
        raise ValueError(f"Corrupted session JSON: {err}") from err
    except (OSError, UnicodeDecodeError) as err:
        raise ValueError(f"Could not read session: {err}") from err
    if not isinstance(document, dict):
        raise ValueError("Session JSON must contain an object.")
    if "version" not in document or "settings" not in document:
        raise ValueError("Session JSON is missing required version or settings keys.")
    if type(document["version"]) is not int:
        raise ValueError("Session version must be an integer.")
    if document["version"] != 1:
        raise ValueError(f"Unknown session version: {document['version']}.")
    if not isinstance(document["settings"], dict):
        raise ValueError("Session settings must be a dictionary.")
    return document["settings"]
