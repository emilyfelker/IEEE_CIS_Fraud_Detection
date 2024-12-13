import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_and_save_roc_curve(model, X_val, y_val, output_path="roc_curve.png"):
    # Predict probabilities
    y_probs = model.predict_proba(X_val)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_val, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save the plot as a PNG file
    plt.savefig(output_path, dpi=300)
    print(f"ROC curve saved to {output_path}")
    plt.close()


def plot_and_save_confusion_matrix(model, X_val, y_val, threshold=0.5, output_path="confusion_matrix.png"):
    # Predict probabilities
    y_probs = model.predict_proba(X_val)[:, 1]

    # Convert probabilities to binary predictions based on threshold
    y_pred = (y_probs >= threshold).astype(int)

    # Compute confusion matrix and normalize to proportions
    cm = confusion_matrix(y_val, y_pred)
    cm_normalized = cm / cm.sum()  # Normalize by total predictions
    cm_labels = ["Not Fraud (0)", "Fraud (1)"]

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title(f"Confusion Matrix (Proportions, Threshold = {threshold})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Save the plot as a PNG file
    plt.savefig(output_path, dpi=300)
    print(f"Confusion matrix plot saved to {output_path}")
    plt.close()


def plot_and_save_feature_importance(model, feature_names, output_path="feature_importance.png", top_n=20):
    # Check for feature importance attribute
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_type = "Feature Importance (Gini/Weight)"
    elif hasattr(model, "coef_"):
        importances = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        importance_type = "Feature Importance (Coefficient)"
    else:
        print("Feature importance not available for this model.")
        return

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Keep only the top N features
    top_features_df = importance_df.head(top_n)

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(top_features_df["Feature"], top_features_df["Importance"], color="skyblue")
    plt.xlabel(importance_type)
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Features by Importance")
    plt.gca().invert_yaxis()  # Invert y-axis for better visualization
    plt.grid(alpha=0.3)

    # Save the plot as a PNG file
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Top {top_n} feature importance plot saved to {output_path}")
    plt.close()