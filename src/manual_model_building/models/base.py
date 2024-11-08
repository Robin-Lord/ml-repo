from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """
    Base class for all models.
    """

    def __init__(self, **kwargs):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def validate_input(self, X: np.ndarray, y: np.ndarray):
        """Validate input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy array")
