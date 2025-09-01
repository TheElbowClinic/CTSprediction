# --------------------------------------------------------------
# 1️⃣  Import Matplotlib
# --------------------------------------------------------------
import matplotlib as mpl

# --------------------------------------------------------------
# 2️⃣  Centralised style dictionary
# --------------------------------------------------------------
STYLE = {
    # ----- Figure / axes ----------
    "axes.linewidth": 0.2,          # frame width
    "lines.linewidth": 0.5,         # line width
    "xtick.labelsize": 2,           # x‑tick font
    "ytick.labelsize": 2,           # y‑tick font
    "axes.titlesize": 4,            # title font size
    "xtick.major.width": 0.3,       # x-tick line width
    "ytick.major.width": 0.3,       # y-tick line width
    "xtick.major.size": 1,          # x-tick length
    "ytick.major.size": 1,          # y-tick length

    # ----- Grid (optional) ----------
    "grid.alpha": 0.1,
    "grid.linestyle": "-",
    "grid.linewidth": 0.2,
}

# --------------------------------------------------------------
# 3️⃣  Apply the style once (executed on import)
# --------------------------------------------------------------
mpl.rcParams.update(STYLE)

# --------------------------------------------------------------
# 4️⃣  If you want a "style‑theme" you can expose a helper
# --------------------------------------------------------------
def apply_custom_style(**kwargs):
    """Update rcParams on the fly."""
    mpl.rcParams.update(kwargs)
