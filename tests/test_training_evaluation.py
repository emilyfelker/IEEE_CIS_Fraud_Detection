import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import ClassifierMixin
from IEEE_CIS_Fraud_Detection.training_evaluation import (
    evaluate_model,
    train_and_evaluate_models,
    get_explicit_hyperparameters,
    choose_model,
    classical_model_evaluation,
    get_top_features
)


@pytest.fixture
def sample_data():
    """Fixture to create sample training and validation data."""
    X_train = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))

    X_val = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'feature3': np.random.rand(50)
    })
    y_val = pd.Series(np.random.randint(0, 2, 50))

    return X_train, X_val, y_train, y_val


def test_evaluate_model(sample_data):
    """Test the evaluate_model function."""
    X_train, X_val, y_train, y_val = sample_data
    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)

    # Act
    result = evaluate_model(model, X_train, X_val, y_train, y_val)

    # Assert
    assert "model" in result
    assert "train_auc" in result
    assert "val_auc" in result
    assert isinstance(result["model"], ClassifierMixin)
    assert isinstance(result["train_auc"], float)
    assert isinstance(result["val_auc"], float)
    assert 0.0 <= result["train_auc"] <= 1.0
    assert 0.0 <= result["val_auc"] <= 1.0


def test_get_explicit_hyperparameters():
    """Test the get_explicit_hyperparameters function."""
    # Arrange
    model = LogisticRegression(max_iter=200, solver='liblinear')

    # Act
    explicit_params = get_explicit_hyperparameters(model)

    # Assert
    assert isinstance(explicit_params, dict)
    assert 'max_iter' in explicit_params
    assert explicit_params['max_iter'] == 200
    assert 'solver' in explicit_params
    assert explicit_params['solver'] == 'liblinear'
    # Parameters not set explicitly should not be in the dict
    assert 'random_state' not in explicit_params  # Assuming default is None


def test_choose_model():
    """Test the choose_model function."""
    # Arrange
    model1 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model2 = XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss')

    # Train models to get AUC scores
    X_train = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))
    X_val = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50)
    })
    y_val = pd.Series(np.random.randint(0, 2, 50))

    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    result1 = evaluate_model(model1, X_train, X_val, y_train, y_val)
    result2 = evaluate_model(model2, X_train, X_val, y_train, y_val)

    models_and_results = [result1, result2]

    # Act
    best_model = choose_model(models_and_results)

    # Determine expected best model based on val_auc
    if result1['val_auc'] > result2['val_auc']:
        expected_best_model = model1
    else:
        expected_best_model = model2

    # Assert
    assert best_model in [model1, model2]
    assert best_model == expected_best_model, "choose_model did not return the model with the highest validation AUC."


def test_get_top_features_logistic(sample_data):
    """Test get_top_features with Logistic Regression."""
    X_train, X_val, y_train, y_val = sample_data
    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)

    feature_names = X_train.columns.tolist()

    # Act
    top_features = get_top_features(model, feature_names, n=2)

    # Assert
    assert isinstance(top_features, list)
    assert len(top_features) == 2
    for feature in top_features:
        assert feature in feature_names


def test_get_top_features_xgboost(sample_data):
    """Test get_top_features with XGBoost."""
    X_train, X_val, y_train, y_val = sample_data
    model = XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    feature_names = X_train.columns.tolist()

    # Act
    top_features = get_top_features(model, feature_names, n=2)

    # Assert
    assert isinstance(top_features, list)
    assert len(top_features) == 2
    for feature in top_features:
        assert feature in feature_names


def test_choose_model_empty():
    """Test choose_model with an empty list."""
    # Arrange
    models_and_results = []

    # Act & Assert
    with pytest.raises(ValueError, match="The input list 'models_and_results' is empty."):
        choose_model(models_and_results)


def test_classical_model_evaluation(sample_data):
    """Test the classical_model_evaluation function."""
    X_train, X_val, y_train, y_val = sample_data

    # Act
    best_model = classical_model_evaluation(X_train, X_val, y_train, y_val)

    # Assert
    assert isinstance(best_model, ClassifierMixin)
    # Optionally, check if best_model is one of the defined models
    assert isinstance(best_model, (LogisticRegression, XGBClassifier))


def test_evaluate_model_invalid_model(sample_data):
    """Test evaluate_model with a non-classifier model."""
    X_train, X_val, y_train, y_val = sample_data

    class NotAClassifier:
        def fit(self, X, y):
            pass

    model = NotAClassifier()

    # Act & Assert
    with pytest.raises(AttributeError):
        evaluate_model(model, X_train, X_val, y_train, y_val)