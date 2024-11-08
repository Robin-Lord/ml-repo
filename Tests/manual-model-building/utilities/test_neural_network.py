"""
Unit tests for the Neural Network implementation.
Tests cover layer initialization, forward pass, and full network operations.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from typing import Callable, List

# Import your classes here
from src.manual_model_building.models.neural_network.neural_network import (
    NeuralNetwork,
    NeuralNetworkLayer,
    LayerDefinition,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Test sigmoid function"""
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """Test ReLU function"""
    return np.maximum(0, x)


class TestNeuralNetworkLayer:
    """Tests for the NeuralNetworkLayer class"""

    def test_layer_initialization(self):
        """Test if layer initializes with correct shapes"""
        neurons, inputs = 3, 4
        layer = NeuralNetworkLayer(
            number_of_neurons=neurons, number_of_inputs=inputs, random_state=42
        )

        assert layer.weights.shape == (neurons, inputs)
        assert layer.biases.shape == (neurons, 1)
        assert callable(layer.activation_function)

    def test_reproducible_initialization(self):
        """Test if random_state produces reproducible results"""
        layer1 = NeuralNetworkLayer(3, 4, random_state=42)
        layer2 = NeuralNetworkLayer(3, 4, random_state=42)

        assert_array_equal(layer1.weights, layer2.weights)
        assert_array_equal(layer1.biases, layer2.biases)

    def test_forward_pass_shape(self):
        """Test if forward pass produces correct output shape"""
        batch_size = 2
        layer = NeuralNetworkLayer(3, 4, random_state=42)
        inputs = np.random.randn(4, batch_size)

        output = layer.forward_pass(inputs)
        assert output.shape == (3, batch_size)

    def test_forward_pass_caching(self):
        """Test if forward pass correctly caches inputs and outputs"""
        layer = NeuralNetworkLayer(3, 4, random_state=42)
        inputs = np.random.randn(4, 1)

        output = layer.forward_pass(inputs)

        assert_array_equal(layer.last_input, inputs)
        assert_array_equal(layer.last_output, output)

    def test_1d_input_handling(self):
        """Test if 1D inputs are correctly reshaped"""
        layer = NeuralNetworkLayer(3, 4, random_state=42)
        inputs_1d = np.random.randn(4)

        output = layer.forward_pass(inputs_1d)
        assert output.shape == (3, 1)

    @pytest.mark.parametrize("activation_fn", [sigmoid, relu])
    def test_different_activation_functions(self, activation_fn: Callable):
        """Test if different activation functions work correctly"""
        layer = NeuralNetworkLayer(
            3, 4, activation_function=activation_fn, random_state=42
        )
        inputs = np.random.randn(4, 1)

        # Manual calculation
        expected = activation_fn(layer.weights @ inputs + layer.biases)
        actual = layer.forward_pass(inputs)

        assert_array_almost_equal(actual, expected)


class TestNeuralNetwork:
    """Tests for the NeuralNetwork class"""

    @pytest.fixture
    def sample_network(self) -> NeuralNetwork:
        """Create a sample network for testing"""
        network = NeuralNetwork(random_state=42)
        layers: List[LayerDefinition] = [
            {"number_of_neurons": 4, "activation_function": relu},
            {"number_of_neurons": 3, "activation_function": sigmoid},
        ]

        input_data = np.random.randn(5, 10)
        # Create the network
        network.sequential_model(input_data=input_data, layers=layers)
        return network

    def test_network_initialization(self, sample_network):
        """Test if network initializes correctly"""
        assert len(sample_network.model_layers) == 2
        assert isinstance(sample_network.model_layers[0], NeuralNetworkLayer)
        assert isinstance(sample_network.model_layers[1], NeuralNetworkLayer)

    def test_network_layer_shapes(self, sample_network):
        """Test if network layers have correct shapes"""
        assert sample_network.model_layers[0].weights.shape == (4, 5)
        assert sample_network.model_layers[1].weights.shape == (3, 4)

    def test_forward_pass_shape(self, sample_network):
        """Test if network forward pass produces correct output shape"""
        batch_size = 2
        inputs = np.random.randn(5, batch_size)
        output = sample_network.forward_pass(inputs)

        assert output.shape == (3, batch_size)

    def test_predict_shape(self, sample_network):
        """Test if predict method handles input/output transposition correctly"""
        samples = 2
        features = 5
        X = np.random.randn(samples, features)
        predictions = sample_network.predict(X)

        assert predictions.shape == (samples, 3)

    def test_no_layers_error(self):
        """Test if network raises error when no layers defined"""
        network = NeuralNetwork()
        with pytest.raises(ValueError):
            network.forward_pass(np.random.randn(5, 1))

    def test_input_shape_validation(self, sample_network):
        """Test if network handles incorrect input shapes"""
        wrong_features = np.random.randn(3, 1)  # Wrong number of features
        with pytest.raises(ValueError, match=r"Input shape mismatch"):
            sample_network.forward_pass(wrong_features)


def test_end_to_end():
    """Test complete network creation and prediction flow"""
    # Create network
    network = NeuralNetwork(random_state=42)
    layers = [
        {"number_of_neurons": 4, "activation_function": relu},
        {"number_of_neurons": 2, "activation_function": sigmoid},
    ]

    # Define input data
    X = np.array([[1, 2, 3], [4, 5, 6]])  # 2 samples, 3 features

    # Create and run model
    network.sequential_model(input_data=X, layers=layers)
    predictions = network.predict(X)

    assert predictions.shape == (2, 2)  # 2 samples, 2 output neurons
    assert np.all((predictions >= 0) & (predictions <= 1))  # Sigmoid output range


def test_reproducibility():
    """Test if networks with same random_state produce identical results"""
    layers = [
        {"number_of_neurons": 4, "activation_function": relu},
        {"number_of_neurons": 2, "activation_function": sigmoid},
    ]

    network1 = NeuralNetwork(random_state=42)
    network2 = NeuralNetwork(random_state=42)

    input_data = np.random.randn(5, 3)

    network1.sequential_model(input_data=input_data, layers=layers)
    network2.sequential_model(input_data=input_data, layers=layers)

    X = np.random.randn(5, 3)

    predictions1 = network1.predict(X)
    predictions2 = network2.predict(X)

    assert_array_almost_equal(predictions1, predictions2)
