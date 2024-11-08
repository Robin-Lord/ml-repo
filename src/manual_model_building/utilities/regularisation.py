"""
Module to hold regularisation functions to be added to models

"""

import numpy as np

from typing import Literal


def regularisation_switchboard(
    weights: np.ndarray,
    regularisation_lambda: float,
    regularisation_function: Literal["l2", "l1"],
) -> np.ndarray:
    """
    Function that takes a series of weights and regularisation parameter, and
    name of the regularisation to use, and returns the weight adjustments to apply

    args:
        weights [np.ndarray]: the existing weights to update
        regularisation_lambda [float]: number to multiply the output by
        regularisation_function [Literal["l2", "l1"]]: whether to use l2 or l1 regularisation

    """

    if regularisation_function == "l2":
        return l2_regularisation(
            weights=weights, regularisation_lambda=regularisation_lambda
        )

    if regularisation_function == "l1":
        return l1_regularisation(
            weights=weights, regularisation_lambda=regularisation_lambda
        )

    # If no function selected just return array of zeros to not modify the weights
    return np.zeros(len(weights))


def l2_regularisation(weights: np.ndarray, regularisation_lambda: float) -> np.ndarray:
    """
    Function that takes a series of weights and a regularisation parameter and
    returns the weights that should be added to the weights in the
    function being modified
    """

    # Apply L2 regularization to all terms except the bias
    l2_term = regularisation_lambda * weights
    l2_term[0] = 0  # Don't regularize the bias term

    return l2_term


def l1_regularisation(weights: np.ndarray, regularisation_lambda: float) -> np.ndarray:
    """
    Function that takes a series of weights and a regularisation parameter and
    returns the weights that should be added to the weights in the function
    being modified
    """

    # Apply L1 regularization to all terms except the bias
    l1_term = regularisation_lambda * np.sign(weights)
    l1_term[0] = 0  # Don't regularize the bias term

    return l1_term
