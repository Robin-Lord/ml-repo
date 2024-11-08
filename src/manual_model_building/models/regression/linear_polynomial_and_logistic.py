"""
Manual implementation of linear, polynomial, and logistic regression.


General principle for linear regression:
    - Small dataset can use closed form solution
        - Fine up to about 10K * 10K
        - Complexity is O(d^3) because of matrix multiplication

    - Larger dataset - need gradient descent
        - Depends on feature number as much as row count because of the
        computational complexity of matrix inversion.
        - Complexity roughly O(nd)

"""

import numpy as np

from typing import Tuple, List, Callable, Optional, Literal, Dict, cast

from ...utilities import (
    generate_polynomials,
    regularisation as utilities_regularisation,
    activation_functions as utilities_activation_functions,
)
from ...visualisation import training_visualisation


# Build these modesl from the base class
from ..base import BaseModel


# Create class that will hold our model and data and handle predictions
class LinearAndLogRegression(BaseModel):
    chosen_polynomial_function: Callable
    learning_rate: float
    n_iterations: int
    polynomial_degree: int
    training_record: Optional[List[Tuple[float, List[float]]]]

    def __init__(
        self,
        working_directory: str,
        learning_rate: float = 1,
        n_iterations: int = 100,
        polynomial_degree: int = 1,
        polynomial_funtion: Callable = generate_polynomials.scikit_learn_polynomial_features,
        regression_type: Literal["linear", "logistic"] = "linear",
        **kwargs,
    ):
        """
        Initialise by establishing methods, don't set data yet so
        we have flexibility over what we train/predict on.

        args:
            working_directory [str]: the folder to save outputs inside
            learning_rate [float]: the learning rate to use in model training
                                optional because can set now or later but will
                                need to set at some point
            n_iterations [float]: the number of iterations to use when fitting the model
                                optional because can set now or later
            polynomial_degree [float]: the degree of polynomial line to fit
                                optional because can set now or later
            polynomial_function [callable]: (default = scikit_learn_polynomial_features) the
                                    function to use to adjust our data to the polynomial degree,
                                    at default using scikit learn's optimised function but can
                                    also use our internal function that accepts the same arguments.
                                    This is always set to some value but we can avoid using polynomials
                                    by setting the polynomial degree to 1
            regression_type (Literal["linear", "logistic"]): (default = "linear) the type of regression
                                    to use, both can work with polynomial features. Linear regression is
                                    for estimating continuous values and logistic is for estimating binary
                                    categories
        """

        super().__init__(**kwargs)

        self.working_directory = working_directory

        # Optionally set learning rate, n_iterations, and polynomial degree
        if learning_rate is not None:
            self.learning_rate = learning_rate

        if n_iterations is not None:
            self.n_iterations = n_iterations

        if polynomial_degree is not None:
            self.polynomial_degree = polynomial_degree

        self.chosen_polynomial_function = polynomial_funtion
        self.regression_type = regression_type

    def polynomial_function(
        self,
        X: np.ndarray,
        polynomial_degree: Optional[int] = None,
        add_bias: Optional[bool] = True,
    ) -> np.ndarray:
        """
        Handler for polynomial function that adjusts our data.

        args:
            X [np.ndarray]: data to adjust
            polynomial_degree [Optional[int]]: (default = 1) degrees of polynomial to add to the data, 1 is stright
                                    if not present will default to object saved
            add_bias [Optiional[bool]]: (default = True) whether to add bias term, defaults to True

        returns:
            the adjusted data, now with added polynomials
        """
        polynomial_degree = self.check_for_value(polynomial_degree, "polynomial_degree")

        return self.chosen_polynomial_function(
            X=X, degree=self.polynomial_degree, add_bias=add_bias
        )

    def check_for_value(self, attribute, attribute_name):
        if attribute is not None:
            return attribute

        if attribute_name in self.__dict__.keys():
            return self.__dict__[attribute_name]

        raise ValueError(
            f"{attribute_name} must either be passed as an argument or set as an attribute"
        )

    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        learning_rate: int = 1,
        n_iterations: int = 100,
        theta: Optional[np.ndarray] = None,
        polynomial_degree: int = 1,
        min_gradient: float = 0.0,
        return_full_train: bool = False,
        regularisation_type: Literal["l2", "l1"] = "l2",
        regularisation_lambda: float = 0.0,
    ) -> Tuple[np.ndarray, List[Tuple[float, np.ndarray]]]:
        """
        Function to handle running of fit function - gathering variables and passing through to
        stand-alone function.


        args:
            X [np.ndarray]: matrix of values to fit the line based on
            y [np.ndarray]: values to attempt to match
            learning_rate [int]: modifier to step-size (also affected by gradient)
            n_iterations [int]: number of steps to take before stopping
            theta [Optional[np.ndarray]]: [bias, weights] (default = None) to be used in calculation
                                    if None then default theta generated in function
            min_gradient [float]: stop training if error gradient gets this low
            return_full_train [bool]: whether to return each prediction line and its calculated cost
                                    WARNING: this could be *very* memory intensive for large
                                    datasets or long training periods. This is mainly
                                    so I can make pretty charts without having to bake it
                                    into the function.
            polynomial_degree [int]: (default = class_set) degrees of polynomial to add to the data, 1 is stright
            polynomial_function [callable]: (default = scikit_learn_polynomial_features) the
                                    function to use to adjust our data to the polynomial degree,
                                    at default using scikit learn's optimised function but can
                                    also use our internal function that accepts the same arguments
            regularisation_type: [Literal["l2", "l1"]] the type of the regularisation to use
            regularisation_lambda: [float] the weight to apply to the regularisation

        returns:
            np.ndarray: bias and weights from final line (should be best-fit)
            [] | list[Tuple(float, np.ndarray)]: list of costs and predicted line for each training step
                                    if return_full_train set to True, see above for warnings

        """

        # Optionally have variables set in class
        X = self.check_for_value(X, "X")
        y = self.check_for_value(y, "y")
        learning_rate = self.check_for_value(learning_rate, "learning_rate")
        n_iterations = self.check_for_value(n_iterations, "n_iterations")
        polynomial_degree = self.check_for_value(polynomial_degree, "polynomial_degree")
        self.polynomial_degree = polynomial_degree

        # Make sure set at object level for later records
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Check which type of regression we're doing (continuous or categorical)
        regression_type = self.regression_type

        training_record: List[Tuple[float, np.ndarray]]
        n_theta: np.ndarray
        normalisation_values: Dict[str, float]

        n_theta, training_record, normalisation_values = fit(
            X=X,
            y=y,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            theta=theta,
            min_gradient=min_gradient,
            return_full_train=return_full_train,
            polynomial_degree=polynomial_degree,
            polynomial_function=self.chosen_polynomial_function,
            regularisation_type=regularisation_type,
            regularisation_lambda=regularisation_lambda,
            regression_type=regression_type,
        )

        self.theta = n_theta
        self.training_record = training_record
        self.normalisation_values = normalisation_values

        return self.theta, self.training_record

    def predict(
        self, X_normalised: Optional[np.ndarray] = None, theta: np.ndarray = None
    ) -> np.ndarray:
        """
        Function to pass through generation of results based on final
        model, either default to final theta, or can be overridden if we want to use
        training thetas

        args:
            X_normalised [np.ndarray]: pre-normalised matrix of values to fit the line based on
            theta [Optional[np.ndarray]]: [bias, weights] (default = None) to be used in calculation
                                    if None then theta set for this object

        returns:
            np.ndarray: the predictionfor X based on theta weights

        """
        X_normalised = self.check_for_value(X_normalised, "X_normalised")
        theta = self.check_for_value(theta, "theta")

        return predict(X_normalised=X_normalised, theta=theta)

    def generate_all_step_predictions(
        self,
        X: Optional[np.ndarray] = None,
        training_record: Optional[List[Tuple[float, List[float]]]] = None,
    ) -> List[np.ndarray]:
        """
        Function to generate a series of predictions, one for each step in training.

        args:
            X [Optional[np.ndarray]]: (default=None) data to run our prediction on
            training_record List[Tuple[float, np.ndarray]]: (default=None) list of
                                    interim training weights to loop through and
                                    generate a prediction on each time


        returns:
            List[np.ndarray]: an ordered list of predictions, one for each
                            set of interim training weights
        """
        X = self.check_for_value(X, "X")
        training_record = self.check_for_value(training_record, "training_record")
        # At this point training_record isn't None
        training_record = cast(List[Tuple[float, List[float]]], training_record)

        prediction_list = []

        # We are not changing our X value so we can just set that as the X_normalised
        # value, if we had new data we would need to pre-normalise
        for gradient, theta in training_record:
            prediction_list.append(
                (gradient, self.predict(X_normalised=X, theta=theta))
            )

        return prediction_list

    def generate_final_and_step_predictions(
        self,
        X: Optional[np.ndarray] = None,
        theta: np.ndarray = None,
        training_record: Optional[List[Tuple[float, List[float]]]] = None,
    ):
        """Function to generate final and step predictions in one

        args:
            X [Optional[np.ndarray]]: (default=None) data to run our prediction on
            theta [Optional[np.ndarray]]: [bias, weights] (default = None) final calculated weights
                                    if None then theta set for this object

            training_record List[Tuple[float, np.ndarray]]: (default=None) list of
                                    interim training weights to loop through and
                                    generate a prediction on each time

        """

        X = self.check_for_value(X, "X")
        theta = self.check_for_value(theta, "theta")
        training_record = self.check_for_value(training_record, "training_record")

        # We are not changing our X value so we can just set that as the X_normalised
        # value, if we had new data we would need to pre-normalise
        final_predicted_line = self.predict(X_normalised=X, theta=theta)
        self.final_predicted_line = final_predicted_line

        step_predictions = self.generate_all_step_predictions(
            X=X, training_record=training_record
        )
        self.training_history = step_predictions
        return final_predicted_line, step_predictions

    def visualise_training(
        self,
        final_predicted_line: Optional[np.ndarray] = None,
        training_history: Optional[List[Tuple[float, np.ndarray]]] = None,
        final_theta: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        original_y: Optional[np.ndarray] = None,
        background_color="black",
        transparent=False,
        axis_col="white",
        log_scale: bool = False,
        figsize: Tuple[int, int] = (24, 16),
        dpi: int = 500,
        training_col="white",
        final_line_col="white",
        og_data_col="gold",
        og_edge_col=None,
        show_training_data=True,
        show_original_data=False,
        show_axes=False,
        y_label: str = "",
        save_name: str = "plot",
    ):
        final_theta = self.check_for_value(final_theta, "theta")
        training_history = self.check_for_value(training_history, "training_history")
        # At this point it should not be None
        training_history = cast(List[Tuple[float, List[float]]], training_history)
        X = self.check_for_value(X, "X")
        original_y = self.check_for_value(original_y, "original_y")
        final_predicted_line = self.check_for_value(
            final_predicted_line, "final_predicted_line"
        )

        # Generate unique save name
        working_dir = "src/experiments/line_progression/loss_lines/outputs"
        full_working_dir = f"{working_dir}/{self.working_directory}"
        visualisation_settings = (
            f"_trd{show_training_data}_ogd{show_original_data}_ax{show_axes}"
        )
        modelling_settings = f"iter{self.n_iterations}_degree{self.polynomial_degree}_rate{self.learning_rate}"

        save_name = f"{full_working_dir}/{modelling_settings}{visualisation_settings}"
        training_visualisation.display_line_fit(
            training_history=training_history,
            final_predicted_line=final_predicted_line,
            X=X,
            original_y=original_y,
            background_color=background_color,
            transparent=transparent,
            axis_col=axis_col,
            log_scale=log_scale,
            figsize=figsize,
            dpi=dpi,
            training_col=training_col,
            final_line_col=final_line_col,
            og_data_col=og_data_col,
            og_edge_col=og_edge_col,
            show_training_data=show_training_data,
            show_original_data=show_original_data,
            show_axes=show_axes,
            y_label=y_label,
            save_name=save_name,
        )


def fit(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: int,
    n_iterations: int,
    theta: Optional[np.ndarray] = None,
    min_gradient: float = 0.0,
    return_full_train: bool = False,
    polynomial_degree: int = 1,
    polynomial_function: Callable = generate_polynomials.scikit_learn_polynomial_features,
    regularisation_type: Literal["l2", "l1"] = "l2",
    regularisation_lambda: float = 0.0,
    regression_type: Literal["linear", "logistic"] = "linear",
) -> Tuple[np.ndarray, List[Tuple[float, np.ndarray]], Dict[str, float]]:
    """
    Function to perform linear or polynomial regression to fit a line based on x input
    and y output.

    Based on gradient descent

    Args:
        X [np.ndarray]: matrix of values to fit the line based on
        y [np.ndarray]: values to attempt to match
        learning_rate [int]: modifier to step-size (also affected by gradient)
        n_iterations [int]: number of steps to take before stopping
        theta [Optional[np.ndarray]]: [bias, weights] (default = None) to be used in calculation
                                if None then default theta generated in function
        min_gradient [float]: stop training if error gradient gets this low
        return_full_train [bool]: whether to return each prediction line and its calculated cost
                                WARNING: this could be *very* memory intensive for large
                                datasets or long training periods. This is mainly
                                so I can make pretty charts without having to bake it
                                into the function.
        polynomial_degree [int]: (default = 1) degrees of polynomial to add to the data, 1 is stright
        polynomial_function [callable]: (default = scikit_learn_polynomial_features) the
                                function to use to adjust our data to the polynomial degree,
                                at default using scikit learn's optimised function but can
                                also use our internal function that accepts the same arguments
        regularisation_type: [Literal["l2", "l1"]] the type of the regularisation to use, can switch
                                off by setting lambda to 0
        regularisation_lambda: [float] the weight to apply to the regularisation

    Returns:
        Tuple[np.ndarray, List[Tuple[float, np.ndarray]], dict]: Final weights (theta),
            list of (gradient magnitude, theta) for each step if return_full_train is True,
            and normalisation features (mean and std).

    """

    # Normalise the data for stability, to avoid wild values, and improve performance
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std  # Normalize X for stability
    # Save features to value for future normalisation
    normalisation_features = {"mean": X_mean, "std": X_std}

    # Get len of data for averaging
    m = len(X)

    # Transform X to a polynomial degree, also add bias
    # term at the start
    X = polynomial_function(X=X, degree=polynomial_degree, add_bias=True)

    # Initialize theta with negative values if not already set
    if theta is None:
        theta = (
            np.random.randn(X.shape[1], 1) * -1
        )  # Random negative initialization for all terms

    iteration = 0
    gradient_magnitude = np.inf

    training_record = []

    while iteration < n_iterations and gradient_magnitude >= min_gradient:
        # Matrix multiplication

        if regression_type == "linear":
            gradient = linear_gradient(theta=theta, m=m, X=X, y=y)
        else:
            gradient = logistic_gradient(theta=theta, m=m, X=X, y=y)

        # Create regularisation weights
        regularisation_weights = utilities_regularisation.regularisation_switchboard(
            weights=theta,
            regularisation_lambda=regularisation_lambda,
            regularisation_function=regularisation_type,
        )

        # Adjust weights

        theta = theta - ((learning_rate * gradient) + regularisation_weights)

        # Get gradient magnitude
        gradient_magnitude = np.linalg.norm(gradient)

        if return_full_train:
            # Make prediction with latest line and save
            training_record.append((gradient_magnitude, theta))

        # Track iteration step
        iteration += 1

    return theta, training_record, normalisation_features


def linear_gradient(
    theta: np.ndarray, m: int, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Function to calculate gradient for linear regression.

    Args:
        theta (np.ndarray): array of weights
        m (int): number of data points
        X (np.ndarray): original input data
        y (np.ndarray): data to predict
    Returns:
        np.ndarray: gradients to use to update the weights
    """
    # Transpose X and matrix multiply by weights to get estimate
    prediction = X.dot(theta)

    # Calculate the difference between our values and actual
    difference = prediction - y

    # Calculate the total gradient of the squared error
    unadjusted_gradient = X.T.dot(difference)

    # Scale error by multiplying by 2 (as part of handling the
    # derivative) and dividing by m (to average so that sample
    # number doesn't affect our cost)
    gradient = (2 / m) * unadjusted_gradient

    return gradient


def logistic_gradient(
    theta: np.ndarray, m: int, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Function to calculate gradient for logistic regression,
    using logarithmic loss (binary cross-entropy)

    Args:
        theta (np.ndarray): array of weights
        m (int): number of data points
        X (np.ndarray): original input data
        y (np.ndarray): data to predict
    Returns:
        np.ndarray: gradients to use to update the weights
    """

    # Calculate prediction
    prediction = X.dot(theta)

    # Put through sigmoid function to always between 0 and 1
    prediction = utilities_activation_functions.sigmoid(prediction)

    # Calculate the loss
    difference = prediction - y

    unadjusted_gradient = X.T.dot(difference)

    # Scale error by multiplying by 2 (as part of handling the
    # derivative) and dividing by m (to average so that sample
    # number doesn't affect our cost)
    gradient = (1 / m) * unadjusted_gradient

    return gradient


def predict(X_normalised, theta):
    """
    Function to use our trained model to make new predictions on new data

    args:
        X_normalised [np.ndarray]: Pre-normalised data that matches the dimensions
                            of our original training data
        theta [np.ndarray]: array of weights for performing the prediction
    """
    return X_normalised.dot(theta)
