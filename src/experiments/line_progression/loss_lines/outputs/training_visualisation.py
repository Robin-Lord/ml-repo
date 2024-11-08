"""

Module to handle generating simple line fit progression charts

"""

from matplotlib import pyplot as plt

from typing import Tuple


def display_line_fit(
    training_history,
    final_predicted_line,
    X,
    original_y,
    background_color="black",
    transparent=False,
    axis_col="white",
    log_scale: bool = False,
    figsize: Tuple[int, int] = (24, 16),
    dpi: int = 500,
    training_col="white",
    final_line_col="white",
    og_data_col="gold",
    show_training_data=True,
    show_original_data=False,
    show_axes=False,
    save_name: str = "plot",
):
    print("Starting vis")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    fig.patch.set_facecolor(background_color)  # Set background color for the figure
    ax.set_facecolor(background_color)  # Set background color for the plot area

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
    y_min, y_max = min(all_y_values), max(all_y_values)
    ax.set_ylim(y_min, y_max)  # Set fixed y-axis limits

    ax.set_xlim(min(X), max(X))  # Set fixed x-axis limits

    # Optionally plot training data
    if show_training_data:
        # Loop through training steps and plot them on the chart
        for cost, prediction in training_history:
            plt.plot(X, prediction, training_col, color=training_col, alpha=0.2)

        plt.plot(
            X,
            final_predicted_line,
            color=final_line_col,
            linewidth=6,
            alpha=0.9,
        )
    # Optionally plot original data
    if show_original_data:
        ax.plot(X, original_y, "o", color=og_data_col, markersize=10)

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
        # plt.legend()
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
