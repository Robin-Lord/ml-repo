"""
Module to handle the building of Neural Network models.

"""

# Standard imports
########################################################
from typing import Callable, TypedDict, List
import numpy as np

from dataclasses import dataclass

# Local imports
########################################################
# Build on top of the base class
from ..base import BaseModel


# Import activation functions
from utilities.activation_functions import sigmoid


@dataclass
class LayerDefinition(TypedDict):
    """Type definition for layer configuration."""

    number_of_neurons: int
    activation_function: Callable


class NeuralNetworkLayer:
    """
    A single layer in a neural network with configurable number of neurons and activation function.

    Attributes:
        activation_function: Callable function to process layer outputs
        number_of_neurons: Number of neurons in this layer
        number_of_inputs: Number of inputs this layer accepts
        weights: Weight matrix of shape (number_of_neurons, number_of_inputs)
        biases: Bias vector of shape (number_of_neurons, 1)
    """

    pass

    def __init__(
        self,
        number_of_neurons: int,
        number_of_inputs: int,
        activation_function: Callable = sigmoid,
        random_state: int = 42,
    ):
        self.activation_function = activation_function
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs = number_of_inputs

        """

        Initialize weights and biases we should have one bias per neuron and
        one weight per input per neuron

        """

        # Use consistent random state initialization
        rng = np.random.RandomState(random_state)

        # Initialize weights using He initialization
        self.weights = rng.randn(number_of_neurons, number_of_inputs) * np.sqrt(
            2.0 / number_of_inputs
        )
        self.biases = rng.randn(number_of_neurons, 1) * 0.01

        # Cache for backpropagation
        self.last_input = None
        self.last_output = None

    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the layer.

        Args:
            inputs: Input data of shape (number_of_inputs, batch_size)

        Returns:
            Layer outputs after activation function
        """

        # Cache input for backpropagation
        self.last_input = inputs

        # Calculate the prediction
        z = np.dot(self.weights, inputs) + self.biases

        # Apply the activation function
        self.last_output = self.activation_function(z)

        return self.last_output


class NeuralNetwork(BaseModel):
    """
    Neural network model supporting sequential layer architecture.

    Attributes:
        random_state: Seed for reproducible weight initialization
        model_layers: List of NeuralNetworkLayer objects
    """

    model_layers: List[NeuralNetworkLayer]
    random_state: int

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.model_layers = []
        super().__init__(**kwargs)

    def sequential_model(
        self, input_data: np.ndarray, layers: List[LayerDefinition]
    ) -> np.ndarray:
        """
        Create a sequential model with specified layer configurations.

        Args:
            input_shape: Number of features in input data
            layers: List of layer definitions specifying neurons and activation functions
        """
        number_of_inputs = input_data.shape[1]  # Based on number of features

        # Create the layers
        self.model_layers = []

        for _layer in layers:
            self.model_layers.append(
                NeuralNetworkLayer(
                    number_of_neurons=_layer["number_of_neurons"],
                    number_of_inputs=number_of_inputs,
                    activation_function=_layer["activation_function"],
                    random_state=self.random_state,
                )
            )
            number_of_inputs = _layer["number_of_neurons"]

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through all layers of the network.

        Args:
            input_data: Input features of shape (number_of_features, batch_size)

        Returns:
            Network predictions after passing through all layers
        """

        for _layer in self.model_layers:
            input_data = _layer.forward_pass(input_data)

        return input_data

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input data.

        Args:
            X: Input features

        Returns:
            Model predictions
        """

        # Have to transpose X because the forward pass
        # expects the input data to be of shape
        # (number_of_features, batch_size)
        return self.forward_pass(X.T).T
