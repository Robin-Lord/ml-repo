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


def generate_polynomial_features(
    x_i: np.array, degree: int, add_bias: bool = True
) -> np.array:
    """
    Function to add polynomial features to an individual vector, up to the
    degree specified in the function call.

    args:
        x_i [np.array]: the vector to add additional values to
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
        feature_combinations = combinations_with_replacement(x_i, d)
        for combo in feature_combinations:
            product = np.prod(combo)
            result.append(product)
    return np.array(result)


def multi_row_polynomial_features(x: np.array, degree: int = 1) -> np.array:
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
    return np.array(list(map(lambda row: generate_polynomial_features(row, degree), x)))
