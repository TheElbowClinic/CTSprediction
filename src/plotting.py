# --------------------------------------------------------------
# 1️⃣  Imports (config is imported for side-effects)
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from . import plot_config     # <-- imports rcParams and defines STYLE

# --------------------------------------------------------------
# 2️⃣  Re-usable plotting function
# --------------------------------------------------------------
def line_plot(
        values,
        labels,
        *,
        x_label="",
        y_label="Correlation value",
        dpi=600,
        figsize=None,
):
    """
    Plots a single-column line graph with a tight, low-font style.

    Parameters
    ----------
    values : array-like
        Y-coordinates.
    labels : array-like
        X-tick labels.
    x_label : str, optional
        Text for the x-axis.
    y_label : str, optional
        Text for the y-axis.
    dpi : int, optional
        Figure resolution.
    figsize : tuple, optional
        (width, height) in inches.
    """
    # 2.1  Create the figure / axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 2.2  Plot the line
    ax.plot(values)

    # 2.3  Set ticks & labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # 2.4  (Optional) Force spines to match the style
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)

    # 2.5  Return the axis for further tweaking
    return ax

# --------------------------------------------------------------
# 3️⃣  Another helper: a scatter plot with the same style
# --------------------------------------------------------------
def scatter_plot(
        x, y,
        *,
        x_label="",
        y_label="",
        dpi=600,
        figsize=None,
        s=20,        # point size
        alpha=0.8,   # point opacity
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x, y, s=s, alpha=alpha)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax





def bar_plot(
        y,                    # bar heights
        labels,           # tick labels
        *,
        x_label="",
        y_label="",
        dpi=600,              # same resolution as your snippet
        figsize=None,         # optional (width, height) in inches
        width=0.8,            # bar width – keep it sensible
        alpha=0.9,            # transparency – the value you used
        align="center",       # mimic your ``align='center'`` line
        edgecolor=None,       # let matplotlib choose
        color=None,           # let matplotlib colour-cycle if None
        hatch=None,           # optional hatch pattern
):
    """
    Plots a bar chart with the *tight*, low-font style you use elsewhere.

    Parameters
    ----------
    y : array-like
        Heights of the bars.
    labels : array-like
        X-tick labels (same length as ``y``).
    x_label, y_label : str, optional
        Axis labels - ``x_label`` defaults to an empty string (as you
        requested).
    dpi : int, optional
        Figure resolution.
    figsize : tuple, optional
        Figure size in inches - if ``None`` the default size is used.
    width : float, optional
        Width of each bar (default 0.8 - the typical value you used).
    alpha : float, optional
        Bar transparency - default 0.9.
    align : {'center', 'edge'}, optional
        Alignment of the bars relative to the ticks (default 'center').
    edgecolor, color : matplotlib colour spec, optional
        Bar face/edge colours.
    hatch : str, optional
        Hatch pattern.
    show : bool, optional
        If ``True`` (default) the plot is shown with ``plt.show()``;
        otherwise the function simply returns the :class:`~matplotlib.axes.Axes`
        instance for further customisation.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plotted axis, ready for additional tweaks.
    """
    # 4.1  Create the figure / axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 4.2  Plot the bars
    x_pos = np.arange(len(labels))
    ax.bar(
        x_pos,
        y,
        align=align,
        alpha=alpha,
        width=width,
        color=color,
        edgecolor=edgecolor if edgecolor is not None else color,
        hatch=hatch,
    )

    # 4.3  Set ticks & labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel(x_label, fontsize=2)
    ax.set_ylabel(y_label)

    # 4.4  (Optional) Force spines to match the tight style
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)

    return ax