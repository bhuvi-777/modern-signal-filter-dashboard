import csv


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
