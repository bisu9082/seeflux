# Shared figure style: pipeline [FIGURE RULES] P1 profile for IEEE TNS (single 3.5 in, double 7.16 in),
# SciencePlots ['science','ieee','no-latex'] first, then the pipeline standard rcParams override it.
import matplotlib
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers styles)

plt.style.use(["science", "ieee", "no-latex"])
matplotlib.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.axisbelow": True,
    "axes.grid": False, "grid.linewidth": 0.5, "grid.alpha": 0.4,
    "legend.frameon": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.prop_cycle": matplotlib.cycler("color", ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]),
    "lines.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    "xtick.top": False, "ytick.right": False,  # spines are hidden, so no ticks on them
})
OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
         "grey": "#595959"}
SINGLE, DOUBLE = 3.5, 7.16

# Paper-wide theme (author choice 2026-10-15: variant A). Data in a blue sequential scale; standard models in
# fixed high-contrast accents; surrogate in black. Used identically in Figs. 1-5.
from matplotlib.colors import LinearSegmentedColormap
DATA_CMAP = LinearSegmentedColormap.from_list("seeflux_blues", ["#a6c8e4", "#5a9fd0", "#2171b5", "#08306b"])
DATA_POINT = "#5a9fd0"
BAND = "#e9eef4"
MODEL = {  # colour, linestyle, label
    "S1_IEC": ("#D55E00", "-", "IEC 62396-1"),
    "S2_JESD_mid": ("#0072B2", "-.", "JESD89 scaling"),
    "S4_AN_ABB": ("#E69F00", (0, (4, 2)), "AN4589 (ABB)"),
    "S4b_AN_CoolCAD": ("#B07800", (0, (1, 1.5)), "AN4589 (CoolCAD)"),
    "FORMULA": ("#111111", "-", "Surrogate, Eq. (1)"),
}


def panel_label(ax, s, dy=-0.27):
    """IEEE Transactions template: 'Labels should appear centered below each subfigure in 8-point Times New Roman
    font in the format of (a) (b) (c).' Regular weight, placed below the x-axis label."""
    ax.text(0.5, dy, s, transform=ax.transAxes, ha="center", va="top", fontsize=8, fontweight="normal",
            fontfamily="Times New Roman")
