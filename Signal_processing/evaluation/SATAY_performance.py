import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sliding_performance import read_change_points, create_overlay_plots, read_data, plot_metric_vs_threshold
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Signal_processing.evaluation.evaluation import (precision, recall, F1_score, annotation_error, 
                       hausdorff_distance, rand_index, adjusted_rand_index,
                       precision_recall_curve, plot_precision_recall_curves,
                       roc_curve_from_cps_by_threshold, mean_absolute_error,
                       match_cps_one_to_one)

setup_plot_style()

CHROMS = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI"]
CPD = {
    "I": [-910, -780, -500, -350, -200, -70, 80, 760],
    "II": [-740, -580, -70, 65, 800],
    "III": [-900, -200, -80, 80, 750],
    "IV": [-900, -800, -740, -500, -380, -60, 60 ],
    "V": [],
    "VI": [],    
    "VII": [],
    "VIII": [],
    "IX": [],
    "X": [],
    "XI": [],
    "XII": [],
    "XIII": [],
    "XIV": [],
    "XV": [],
    "XVI": [],
}

output_folder = "Signal_processing/results/sliding_cpd_performance/SATAY_CPD/"
# make folder if not exists 
os.makedirs(output_folder, exist_ok=True)

