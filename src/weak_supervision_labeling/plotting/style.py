# src/weak_supervision_labeling/plotting/style.py

from __future__ import annotations
import matplotlib as mpl

def set_plot_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "CMU Serif"],
        "figure.titlesize": 11,
        "figure.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.titleweight": "normal",
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "figure.autolayout": False,
    })