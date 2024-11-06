"""

Module to handle generating simple line fit progression charts

"""

from matplotlib import pyplot as plt

from typing import List, Optional, Tuple


def display_line_fit(
    training_history: List[Tuple[float, List[float]]],
    final_predicted_line: List[float],
    X: List[float],
    original_y: List[float],
    background_color: str = "black",
    transparent: bool = False,
    axis_col: str = "white",
    log_scale: bool = False,
    figsize: Tuple[int, int] = (24, 16),
    dpi: int = 500,
    training_col: str = "white",
    final_line_col: str = "white",
    og_data_col: str = "gold",
    og_edge_col: Optional[str] = None,
    show_training_data: bool = True,
    show_original_data: bool = False,
    show_axes: bool = False,
    y_label: str = "",
    save_name: str = "plot",
) -> None:
    """
    Plot and save a visualization of the training process for a line-fitting model.

    Args:
        training_history (List[Tuple[float, List[float]]]): Training steps with cost and predictions.
        final_predicted_line (List[float]): Final line predictions after training.
        X (List[float]): X-axis values for plotting.
        original_y (List[float]): Original y-axis values for comparison.
        background_color (str): Color of the plot background. Default is "black".
        transparent (bool): If True, saves the plot with a transparent background. Default is False.
        axis_col (str): Color of the plot axes. Default is "white".
        log_scale (bool): If True, sets the y-axis to logarithmic scale. Default is False.
        figsize (Tuple[int, int]): Size of the figure. Default is (24, 16).
        dpi (int): Resolution of the saved plot. Default is 500.
        training_col (str): Color of the training data line. Default is "white".
        final_line_col (str): Color of the final line. Default is "white".
        og_data_col (str): Color of the original data points. Default is "gold".
        show_training_data (bool): Whether to plot training data lines. Default is True.
        show_original_data (bool): Whether to plot the original data points. Default is False.
        show_axes (bool): Whether to display axes and labels. Default is False.
        y_label (str): Visible label to show on y-axis to explain what it charts
        save_name (str): Name to save the plot file as. Default is "plot".

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    fig.patch.set_facecolor(background_color)  # Set background color for the figure
    ax.set_facecolor(background_color)  # Set background color for the plot area
    # Increase axis number font size
    ax.tick_params(axis="both", which="major", labelsize=20)  # Adjust size as needed

    if log_scale:
        ax.set_yscale(
            "log"
        )  # Optional: Set y-axis to logarithmic scale for visual effect

    # Gather all y-values from the data for consistent axis scaling
    all_y_values = (
        [y for _, prediction in training_history for y in prediction]
        + list(final_predicted_line)
        + list(original_y)
    )
    # Make sure that min value is not negatively infinite
    y_min = float(max(0, min(all_y_values)))
    y_max = float(max(all_y_values))

    ax.set_ylim(y_min, y_max)  # Set fixed y-axis limits

    ax.set_xlim(min(X), max(X))  # Set fixed x-axis limits

    # Optionally plot training data
    if show_training_data:
        # Loop through training steps and plot them on the chart
        for cost, prediction in training_history:
            plt.plot(X, prediction, color=training_col, alpha=0.2)

        plt.plot(
            X,
            final_predicted_line,
            color=final_line_col,
            linewidth=6,
            alpha=0.9,
        )
    # Optionally plot original data
    if show_original_data:
        # Optionally add border to datapoints for visibility
        datapoint_style_args = {
            "markerfacecolor": og_data_col,
            "markersize": 10,
        }
        if og_edge_col is not None:
            datapoint_style_args["markeredgecolor"] = og_edge_col
        ax.plot(X, original_y, "o", **datapoint_style_args)

    # Optionally show axes
    if show_axes:
        for spine in ax.spines.values():
            spine.set_visible(True)  # Make all axis spines visible
        ax.tick_params(
            left=True,
            bottom=True,
            labelleft=True,
            labelbottom=True,
            color=axis_col,
            labelcolor=axis_col,
        )  # Show ticks and labels
        # Add y-axis label
        if y_label != "":
            ax.set_ylabel(y_label, fontsize=22, color=axis_col)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)  # Hide all axis spines
        ax.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )  # Hide ticks and labels
        plt.gca().set_position([0, 0, 1, 1])  # Remove margins around the plot

    # Save each plot with a unique filename
    plt.savefig(f"{save_name}.png", transparent=transparent)
    plt.close(fig)  # Close the figure after saving to avoid overlap in memory
