import csv


def save_signal_csv(path, t, original, filtered):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time (s)", "Original", "Filtered"])
        for row in zip(t, original, filtered):
            writer.writerow(row)


def save_plot(path, figure):
    figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())
