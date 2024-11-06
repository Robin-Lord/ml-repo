"""
Module to make it easier to extend data into the past and future
"""

import numpy as np


def extend_data(X: np.array, past_extension: int, future_extension: int):
    """
    Take existing simple 2D array, adjust the past and future numbers to
    allow lines to be projected before and after data.
    """
    return np.arange(X.min() - past_extension, X.max() + future_extension).reshape(
        -1, 1
    )
