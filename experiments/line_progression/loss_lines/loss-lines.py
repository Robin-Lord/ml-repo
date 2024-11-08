import importlib  # Handle file changes

# Manage arrays of numbers
import itertools  # Combine lists into combinations

# Internal imports
from manual_model_building.models.regression import linear_polynomial_and_logistic

from experiments.line_progression.loss_lines.data.load import (
    gini_index,
    owid_co2,
    un_life_expectancy,
)


# Reload the modules to pick up changes
importlib.reload(linear_polynomial_and_logistic)
importlib.reload(gini_index)


CURRENT_DIRECTORY = "experiments/line_progression/loss_lines"
OUTPUT_DIRECTORY = f"{CURRENT_DIRECTORY}/visualisation"


def run_for_gini():
    """
    Run polynomial regression visualization for US Gini Index data.

    Fits a polynomial regression model (degree 2) to US Gini Index data from 1963-2022.
    Creates three visualization variants:
    1. Training progression with original data and axes
    2. Training progression with final prediction only
    3. Original data and final prediction only

    The visualizations use a black background with white/gold styling to show:
    - Model training progression
    - Final polynomial fit
    - Original data points

    Model parameters:
    - Polynomial degree: 2
    - Learning rate: 0.05
    - Training iterations: 500
    """
    gini_index_data = gini_index.fetch_gini_index_data()

    # Separate data
    X = gini_index_data[["Year"]].values
    y = gini_index_data[["Gini Index"]].values

    # Set polynomial degree to fit
    degrees = [2]

    # Gradient Descent settings
    learning_rates = [0.05]  # Adjust as needed for convergence
    n_iterationses = [500]

    for combination in itertools.product(degrees, learning_rates, n_iterationses):
        degree = combination[0]
        learning_rate = combination[1]
        n_iterations = combination[2]

        _model = linear_polynomial_and_logistic.LinearAndLogRegression(
            working_directory="gini_3",
            learning_rate=learning_rate,
            polynomial_degree=degree,
            n_iterations=n_iterations,
        )
        print(f"{degree}, {learning_rate}, {n_iterations}")

        # Fit the model
        _model.fit(X=X, y=y, return_full_train=True)

        # Update data-for-prediction to match adjustment in training data
        X_poly = _model.polynomial_function(X=X)

        # Generate predictions
        final_prediction = _model.predict(X=X_poly)

        # Generate and save our final prediction, and one for each step
        _model.generate_final_and_step_predictions(X=X_poly)

        # Create different versions of chart
        for settings in [(True, True), (True, False), (False, True)]:
            show_training_data = settings[0]
            show_original_data = settings[1]
            show_show_axes = settings[1]

            _model.visualise_training(
                X=X.flatten(),
                original_y=y,
                final_predicted_line=final_prediction,
                show_training_data=show_training_data,
                show_original_data=show_original_data,
                show_axes=show_show_axes,
                background_color="black",
                transparent=True,
                axis_col="white",
                training_col="white",
                final_line_col="gold",
                og_data_col="gold",
                og_edge_col="white",
                y_label="Wealth gap (GINI index)",
            )


def run_for_carbon():
    """
    Run polynomial regression visualization for global CO2 emissions data.

    Fits a polynomial regression model (degree 3) to historical CO2 emissions data
    and creates three visualization variants:
    1. Training progression with final prediction and original data
    2. Training progression with final prediction only
    3. Training progression with original data only

    The visualizations use a white background with black lines to emphasize the
    emissions trend over time.

    Model parameters:
        - Polynomial degree: 3
        - Learning rate: 0.005
        - Training iterations: 1000
    """
    X, y = owid_co2.fetch_global_co2_emissions_data()

    # Set polynomial degree to fit
    degrees = [3]

    # Gradient Descent settings
    learning_rates = [0.005]  # Adjust as needed for convergence
    n_iterationses = [1000]

    for combination in itertools.product(degrees, learning_rates, n_iterationses):
        degree = combination[0]
        learning_rate = combination[1]
        n_iterations = combination[2]

        _model = linear_polynomial_and_logistic.LinearAndLogRegression(
            working_directory="carbon",
            learning_rate=learning_rate,
            polynomial_degree=degree,
            n_iterations=n_iterations,
        )
        print(f"{degree}, {learning_rate}, {n_iterations}")

        # Fit the model
        _model.fit(X=X, y=y, return_full_train=True)

        # Update data-for-prediction to match adjustment in training data
        X_poly = _model.polynomial_function(X=X)

        # Generate predictions
        final_prediction = _model.predict(X=X_poly)

        # Generate and save our final prediction, and one for each step
        _model.generate_final_and_step_predictions(X=X_poly)

        # Create different versions of chart
        for settings in [(True, True), (True, False), (False, True)]:
            show_training_data = settings[0]
            show_original_data = settings[1]
            show_show_axes = settings[1]

            _model.visualise_training(
                X=X.flatten(),
                original_y=y,
                final_predicted_line=final_prediction,
                show_training_data=show_training_data,
                show_original_data=show_original_data,
                show_axes=show_show_axes,
                background_color="white",
                transparent=True,
                axis_col="black",
                training_col="black",
                final_line_col="black",
                og_data_col="black",
                og_edge_col="grey",
                y_label="Annual tonnes of CO2",
            )


def run_for_life_expectancy():
    """
    Run polynomial regression visualization for global life expectancy data.

    Fits a polynomial regression model (degree 2) to UN life expectancy data and creates
    three visualization variants:
    1. Training progression with original data and axes
    2. Training progression only
    3. Training progression with final prediction line

    The visualizations show the model's evolution during training using red/white lines
    on a black background. Training uses gradient descent with:
    - Learning rate: 0.03
    - Iterations: 1000
    - No regularization (for aesthetic purposes)
    """
    X, y = un_life_expectancy.fetch_life_expectancy_data()

    # Set polynomial degree to fit
    degrees = [2]

    # Gradient Descent settings
    learning_rates = [0.03]  # Adjust as needed for convergence
    n_iterationses = [1000]

    for combination in itertools.product(degrees, learning_rates, n_iterationses):
        degree = combination[0]
        learning_rate = combination[1]
        n_iterations = combination[2]

        _model = linear_polynomial_and_logistic.LinearAndLogRegression(
            working_directory="life_expectancy",
            learning_rate=learning_rate,
            polynomial_degree=degree,
            n_iterations=n_iterations,
        )
        print(f"{degree}, {learning_rate}, {n_iterations}")

        # Fit the model
        # No regularisation because without it, it looks better
        _model.fit(
            X=X,
            y=y,
            return_full_train=True,
        )

        # Update data-for-prediction to match adjustment in training data
        X_poly = _model.polynomial_function(X=X)

        # Generate predictions
        final_prediction = _model.predict(X_normalised=X_poly)

        # Generate and save our final prediction, and one for each step
        _model.generate_final_and_step_predictions(X=X_poly)

        # Create different versions of chart
        for settings in [(True, True), (True, False), (False, True)]:
            show_training_data = settings[0]
            show_original_data = settings[1]
            show_show_axes = settings[1]

            # Remove the first few steps from the training for our vis
            training_history_for_vis = _model.training_history[100:]

            _model.visualise_training(
                X=X.flatten(),
                original_y=y,
                training_history=training_history_for_vis,
                final_predicted_line=final_prediction,
                show_training_data=show_training_data,
                show_original_data=show_original_data,
                show_axes=show_show_axes,
                background_color="black",
                transparent=True,
                axis_col="white",
                training_col="red",
                final_line_col="white",
                og_data_col="white",
                og_edge_col="red",
                y_label="Life expectancy of newborns",
            )


if __name__ == "__main__":
    # run_for_gini()
    # run_for_carbon()
    run_for_life_expectancy()
