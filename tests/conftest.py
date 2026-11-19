import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib

matplotlib.use("Agg")
