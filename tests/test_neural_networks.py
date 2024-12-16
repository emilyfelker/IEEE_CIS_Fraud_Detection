import pytest
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from unittest.mock import patch
from IEEE_CIS_Fraud_Detection.neural_networks import build_neural_network, train_neural_networks


@pytest.fixture
def mock_data():
    X_train = pd.DataFrame(np.random.rand(100, 10))  # 100 samples, 10 features
    X_val = pd.DataFrame(np.random.rand(20, 10))     # 20 samples, 10 features
    y_train = pd.Series(np.random.randint(0, 2, 100))  # Binary target
    y_val = pd.Series(np.random.randint(0, 2, 20))     # Binary target
    return X_train, X_val, y_train, y_val


@pytest.mark.parametrize("gelu_layer_sizes,last_layer", [
    ([32], 16),
    ([64, 32], 8),
    ([128], 64)
])
def test_build_neural_network_with_different_architectures(gelu_layer_sizes, last_layer):
    # Arrange: Set input dimensions
    input_dim = 10  # Example input dimension

    # Act: Build the model with the given architecture
    model = build_neural_network(input_dim, gelu_layer_sizes, last_layer)

    # Build the model to initialize layers (optional, depending on Keras version)
    model.build(input_shape=(None, input_dim))

    # Assert: Check if the model is a Sequential model with expected layers and input shape
    assert isinstance(model, Sequential), "Model is not a Keras Sequential model."
    assert len(model.layers) > 0, "Model has no layers."

    # Assert the model's input shape
    assert model.input_shape == (None, input_dim), f"Expected input shape {(None, input_dim)}, got {model.input_shape}."

    # Assert that the output layer has the correct activation
    assert model.layers[-1].activation.__name__ == 'sigmoid', "Output layer does not have 'sigmoid' activation."

    # Assert there are Dense layers with 'gelu' activation
    gelu_layers_present = any(
        isinstance(layer, Dense) and layer.activation.__name__ == 'gelu'
        for layer in model.layers
    )
    assert gelu_layers_present, "No Dense layers with 'gelu' activation found."

    # Assert there are Dropout layers
    dropout_layers_present = any(
        isinstance(layer, Dropout)
        for layer in model.layers
    )
    assert dropout_layers_present, "No Dropout layers found."

    # Check the number of GELU and Dropout layers based on gelu_layer_sizes
    expected_gelu_layers = len(gelu_layer_sizes) + 1  # +1 for the final hidden Dense layer
    actual_gelu_layers = sum(
        1 for layer in model.layers
        if isinstance(layer, Dense) and layer.activation.__name__ == 'gelu'
    )
    assert actual_gelu_layers == expected_gelu_layers, (
        f"Expected {expected_gelu_layers} GELU Dense layers, found {actual_gelu_layers}."
    )

    expected_dropout_layers = len(gelu_layer_sizes)
    actual_dropout_layers = sum(
        1 for layer in model.layers
        if isinstance(layer, Dropout)
    )
    assert actual_dropout_layers == expected_dropout_layers, (
        f"Expected {expected_dropout_layers} Dropout layers, found {actual_dropout_layers}."
    )


def test_build_neural_network():
    # Arrange: Set input dimensions
    input_dim = 10
    gelu_layer_sizes = [32, 64]
    last_layer = 16

    # Act: Build the model
    model = build_neural_network(input_dim, gelu_layer_sizes, last_layer)

    # Build the model to initialize layers
    model.build(input_shape=(None, input_dim))

    # Assert: Check if the model is a Sequential model with expected layers and input shape
    assert isinstance(model, Sequential), "Model is not a Keras Sequential model."
    assert len(model.layers) > 0, "Model has no layers."

    # Assert the model's input shape
    assert model.input_shape == (None, input_dim), f"Expected input shape {(None, input_dim)}, got {model.input_shape}."

    # Assert that the output layer has the correct activation
    assert model.layers[-1].activation.__name__ == 'sigmoid', "Output layer does not have 'sigmoid' activation."

    # Assert there are Dense layers with 'gelu' activation
    gelu_layers_present = any(
        isinstance(layer, Dense) and layer.activation.__name__ == 'gelu'
        for layer in model.layers
    )
    assert gelu_layers_present, "No Dense layers with 'gelu' activation found."

    # Assert there are Dropout layers
    dropout_layers_present = any(
        isinstance(layer, Dropout)
        for layer in model.layers
    )
    assert dropout_layers_present, "No Dropout layers found."

    # Check the number of GELU and Dropout layers based on gelu_layer_sizes
    expected_gelu_layers = len(gelu_layer_sizes) + 1  # +1 for the final hidden Dense layer
    actual_gelu_layers = sum(
        1 for layer in model.layers
        if isinstance(layer, Dense) and layer.activation.__name__ == 'gelu'
    )
    assert actual_gelu_layers == expected_gelu_layers, (
        f"Expected {expected_gelu_layers} GELU Dense layers, found {actual_gelu_layers}."
    )

    expected_dropout_layers = len(gelu_layer_sizes)
    actual_dropout_layers = sum(
        1 for layer in model.layers
        if isinstance(layer, Dropout)
    )
    assert actual_dropout_layers == expected_dropout_layers, (
        f"Expected {expected_dropout_layers} Dropout layers, found {actual_dropout_layers}."
    )


def test_train_neural_networks(mock_data):
    # Arrange: Get mock data
    X_train, X_val, y_train, y_val = mock_data

    # Define a side effect function for predict
    def mock_predict_side_effect(input_data):
        return np.random.rand(len(input_data))  # Return a random array matching input length

    # Act: Mock training process
    with patch('IEEE_CIS_Fraud_Detection.neural_networks.Sequential.fit') as mock_fit, \
         patch('IEEE_CIS_Fraud_Detection.neural_networks.Sequential.predict') as mock_predict, \
         patch.object(Sequential, 'summary') as mock_summary:  # To reduce unnecessary output while testing

        mock_fit.return_value = None  # Mock training completion
        mock_predict.side_effect = mock_predict_side_effect  # Assign the side effect

        trained_models = train_neural_networks(X_train, X_val, y_train, y_val)

    # Assert: Check that models were trained
    assert len(trained_models) == 6, f"Expected 6 trained models, got {len(trained_models)}."
    for idx, (model, train_auc, val_auc) in enumerate(trained_models, 1):
        assert isinstance(model, Sequential), f"Model {idx} is not a Keras Sequential model."
        assert isinstance(train_auc, float), f"Train AUC for model {idx} is not a float."
        assert isinstance(val_auc, float), f"Validation AUC for model {idx} is not a float."
        assert 0 <= train_auc <= 1, f"Train AUC for model {idx} is out of bounds: {train_auc}."
        assert 0 <= val_auc <= 1, f"Validation AUC for model {idx} is out of bounds: {val_auc}."