import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from IEEE_CIS_Fraud_Detection.data_visualization import (
    plot_and_save_roc_curve,
    plot_and_save_confusion_matrix,
    plot_and_save_feature_importance
)


# Custom Mock Model Class
class MockModel:
    def __init__(self, predict_proba_return=None, feature_importances_=None, coef_=None):
        self.predict_proba = Mock(return_value=predict_proba_return)
        if feature_importances_ is not None:
            self.feature_importances_ = feature_importances_
        if coef_ is not None:
            self.coef_ = coef_


# Fixtures
@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_data():
    X_val = pd.DataFrame(np.random.rand(20, 5), columns=[f"feature_{i}" for i in range(5)])
    y_val = pd.Series(np.random.randint(0, 2, 20))
    return X_val, y_val


@pytest.fixture
def feature_names():
    return [f"feature_{i}" for i in range(5)]


def test_plot_and_save_roc_curve(mock_model, mock_data):
    X_val, y_val = mock_data

    # Mock predict_proba to return fixed probabilities
    mock_model.predict_proba.return_value = np.array([
        [0.3, 0.7],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.5, 0.5],
        [0.1, 0.9],
        [0.4, 0.6],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.35, 0.65],
        [0.55, 0.45],
        [0.15, 0.85],
        [0.45, 0.55],
        [0.25, 0.75],
        [0.65, 0.35],
        [0.05, 0.95],
        [0.75, 0.25],
        [0.85, 0.15],
        [0.95, 0.05],
        [0.05, 0.95],
        [0.15, 0.85],
    ])

    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        with patch('matplotlib.pyplot.close') as mock_close:
            plot_and_save_roc_curve(mock_model, X_val, y_val, output_path="test_output/roc_curve_test.png")

    # Assert predict_proba was called correctly
    mock_model.predict_proba.assert_called_once_with(X_val)

    # Assert savefig was called with the correct path
    mock_savefig.assert_called_once_with("test_output/roc_curve_test.png", dpi=300)

    # Assert plt.close was called at least once
    assert mock_close.call_count >= 1, "plt.close was not called."


def test_plot_and_save_confusion_matrix(mock_model, mock_data):
    X_val, y_val = mock_data

    # Mock predict_proba to return fixed probabilities
    mock_model.predict_proba.return_value = np.array([
        [0.3, 0.7],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.5, 0.5],
        [0.1, 0.9],
        [0.4, 0.6],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.35, 0.65],
        [0.55, 0.45],
        [0.15, 0.85],
        [0.45, 0.55],
        [0.25, 0.75],
        [0.65, 0.35],
        [0.05, 0.95],
        [0.75, 0.25],
        [0.85, 0.15],
        [0.95, 0.05],
        [0.05, 0.95],
        [0.15, 0.85],
    ])

    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        with patch('matplotlib.pyplot.close') as mock_close:
            plot_and_save_confusion_matrix(
                mock_model,
                X_val,
                y_val,
                threshold=0.5,
                output_path="test_output/confusion_matrix_test.png"
            )

    # Assert predict_proba was called correctly
    mock_model.predict_proba.assert_called_once_with(X_val)

    # Assert savefig was called with the correct path
    mock_savefig.assert_called_once_with("test_output/confusion_matrix_test.png", dpi=300)

    # Assert plt.close was called at least once
    assert mock_close.call_count >= 1, "plt.close was not called."


def test_plot_and_save_feature_importance_with_feature_importances(mock_model, feature_names):
    # Mock feature_importances_
    mock_model.feature_importances_ = np.array([0.2, 0.3, 0.1, 0.25, 0.15])

    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        with patch('matplotlib.pyplot.close') as mock_close:
            plot_and_save_feature_importance(
                model=mock_model,
                feature_names=feature_names,
                output_path="test_output/feature_importance_test.png",
                top_n=3
            )

    # Assert savefig was called with the correct path
    mock_savefig.assert_called_once_with("test_output/feature_importance_test.png", dpi=300, bbox_inches="tight")

    # Assert plt.close was called once
    mock_close.assert_called_once()


def test_plot_and_save_feature_importance_with_coef(mock_model, feature_names):
    # Mock coef_
    mock_model.coef_ = np.array([[0.4, -0.2, 0.1, 0.3, -0.1]])

    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        with patch('matplotlib.pyplot.close') as mock_close:
            plot_and_save_feature_importance(
                model=mock_model,
                feature_names=feature_names,
                output_path="test_output/feature_importance_coef_test.png",
                top_n=2
            )

    # Assert savefig was called with the correct path
    mock_savefig.assert_called_once_with("test_output/feature_importance_coef_test.png", dpi=300, bbox_inches="tight")

    # Assert plt.close was called once
    mock_close.assert_called_once()


def test_plot_and_save_feature_importance_no_importance(mock_model, feature_names, capsys):
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        with patch('matplotlib.pyplot.close') as mock_close:
            plot_and_save_feature_importance(
                model=mock_model,
                feature_names=feature_names,
                output_path="test_output/feature_importance_none_test.png",
                top_n=5
            )

    # Assert savefig was not called since feature importance is unavailable
    mock_savefig.assert_not_called()

    # Assert plt.close was not called
    mock_close.assert_not_called()

    # Capture printed output
    captured = capsys.readouterr()
    assert "Feature importance not available for this model." in captured.out