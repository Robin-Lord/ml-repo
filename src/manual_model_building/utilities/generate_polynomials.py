"""

Manual implementation of generating polynomial features in a matrix.

Mainly a thought exercise - this is done more efficiently by SKLearns.PolynomialFeatures

    - multi_row_polynomial_features
    - generate_polynomial_features

"""

import numpy as np


# Create every combination of items in a list as if they were
# drawn with replacement (including combinations of items with themselves)
from itertools import combinations_with_replacement

# Use scikit learn's optimised polynomial features generation
from sklearn.preprocessing import PolynomialFeatures  # for typing


def generate_polynomial_features(
    X_i: np.array, degree: int, add_bias: bool = True
) -> np.array:
    """
    Function to add polynomial features to an individual vector, up to the
    degree specified in the function call.

    args:
        X_i [np.array]: the vector to add additional values to
        degree [int]: the degree of polynomial to add
        add_bias [bool]: (default = True) whether to add bias term at start

    returns:
        [np.array]: updated vector with new polynomial features
    """

    # Start with empty array, or with added bias if needed
    result = []
    if add_bias:
        result.append(1)

    # Loop through each degree to add, for each - extract n-choose-d
    # features, with replacement, and multiply them together, then
    # append the answer
    for d in range(1, degree + 1):
        feature_combinations = combinations_with_replacement(X_i, d)
        for combo in feature_combinations:
            product = np.prod(combo)
            result.append(product)
    return np.array(result)


def multi_row_polynomial_features(X: np.array, degree: int = 1) -> np.array:
    """
    Function to apply generate_polynomial_features to an entire np matrix.

    This isn't as efficient as SKLearn's PolynomialFeatures so it's more of a
    thought exercise than anything.

    args:
        x [np.array]: matrix of np elements to apply the polynomial function to
        degree [int]: the number of degrees of polynomial to apply

    returns:
        np.array: the updated matrix with the polynomials added

    """

    # Normalise the data for stability, to avoid wild values, and improve performance
    X = (X - X.mean()) / X.std()

    return np.array(list(map(lambda row: generate_polynomial_features(row, degree), X)))


def scikit_learn_polynomial_features(X: np.array, degree: int, add_bias: bool = True):
    """
    Easy implementation of scikit learn's polynomial features.

    Must have parity of arguments with internal function, for easy
    swappability

    args:
        X [np.array]: the vector to add additional values to
        degree [int]: the degree of polynomial to add
        add_bias [bool]: (default = True) whether to add bias term at start
    """

    # Normalise the data for stability, to avoid wild values, and improve performance
    X = (X - X.mean()) / X.std()

    # Set settings for feature generation
    poly = PolynomialFeatures(degree=degree, include_bias=add_bias)

    # Input needs to be 2D array - if x is 1D, reshape it
    x_2d = X.reshape(-1, 1) if X.ndim == 1 else X

    # Add polynomial features to data and return
    return poly.fit_transform(x_2d)
