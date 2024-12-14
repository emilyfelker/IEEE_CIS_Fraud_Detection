import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def evaluate_model(model, X_train, X_val, y_train, y_val):
    # Predict probabilities
    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]

    # Calculate AUC
    train_auc = roc_auc_score(y_train, train_probs)
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"Model: {model.__class__.__name__}")
    print(f"  Training AUC: {train_auc:.4f}")
    print(f"  Validation AUC: {val_auc:.4f}")
    return {"model": model, "train_auc": train_auc, "val_auc": val_auc}


def train_and_evaluate_models(models, X_train, X_val, y_train, y_val):
    models_and_results = []
    for model in models:
        explicit_hyperparameters = get_explicit_hyperparameters(model)
        print(f"Training model: {model.__class__.__name__} with {explicit_hyperparameters}")
        trained_model = model.fit(X_train, y_train)
        result = evaluate_model(trained_model, X_train, X_val, y_train, y_val)
        models_and_results.append({
            "model": trained_model,
            "model_name": trained_model.__class__.__name__,
            "hyperparameters": explicit_hyperparameters,
            "train_auc": result["train_auc"],
            "val_auc": result["val_auc"]
        })
    return models_and_results


def get_explicit_hyperparameters(model):
    # Retrieve the default parameters for the model's class
    default_params = type(model)().get_params()

    # Compare with the current model's parameters
    current_params = model.get_params()

    # Keep only the parameters that differ from the defaults
    explicit_params = {k: v for k, v in current_params.items() if v != default_params[k]}
    return explicit_params


def choose_model(models_and_results):
    if not models_and_results:
        raise ValueError("The input list 'models_and_results' is empty.")

    # Find the model with the highest validation AUC
    best_model_entry = max(models_and_results, key=lambda x: x["val_auc"])
    return best_model_entry["model"]


def classical_model_evaluation(X_train, X_val, y_train, y_val, feature_names):
    # Define models
    models = [
        LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
        XGBClassifier(n_estimators = 32, max_depth = 4),
        XGBClassifier(n_estimators = 64, max_depth = 4),
        XGBClassifier(n_estimators = 32, max_depth = 8), # best so far

    ]

    # Train and evaluate models
    models_and_results = train_and_evaluate_models(models, X_train, X_val, y_train, y_val)
    best_model = choose_model(models_and_results)

    return best_model


def get_top_features(model, feature_names, n=32):
    # Check model type and extract feature importance
    if hasattr(model, "coef_"):  # Logistic Regression
        # Extract absolute values of coefficients for importance
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):  # XGBoost
        # Use feature_importances_ directly
        importances = model.feature_importances_
    else:
        raise ValueError("Model type not supported. Provide a Logistic Regression or XGBoost model.")

    # Pair feature names with their importances
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

    # Sort by importance and get the top n features
    top_features = (
        feature_importance_df
        .sort_values(by="Importance", ascending=False)
        .head(n)["Feature"]
        .tolist()
    )
    return top_features
