import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# TODO: Generate ipython notebook for submission to competition
# TODO: Write ReadMe file for GitHub


def load_data(zip_file_path, n_rows=None):
    try: # for local environment
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open('train_transaction.csv') as train_transaction:
                train_transaction = pd.read_csv(train_transaction, index_col='TransactionID', nrows=n_rows)
            with z.open('train_identity.csv') as train_identity:
                train_identity = pd.read_csv(train_identity, index_col='TransactionID', nrows=n_rows)
            with z.open('test_transaction.csv') as test_transaction:
                test_transaction = pd.read_csv(test_transaction, index_col='TransactionID', nrows=n_rows)
            with z.open('test_identity.csv') as test_identity:
                test_identity = pd.read_csv(test_identity, index_col='TransactionID', nrows=n_rows)
    except (FileNotFoundError, zipfile.BadZipFile): # for Kaggle environment
        train_transaction = pd.read_csv(
            '/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID', nrows=n_rows)
        train_identity = pd.read_csv(
            '/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID', nrows=n_rows)
        test_transaction = pd.read_csv(
            '/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID', nrows=n_rows)
        test_identity = pd.read_csv(
            '/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID', nrows=n_rows)
    print("CSV files read in.")

    # Normalize column names (because it turns out the test dataframes use different naming conventions)
    train_transaction.columns = train_transaction.columns.str.replace('-', '_')
    train_identity.columns = train_identity.columns.str.replace('-', '_')
    test_transaction.columns = test_transaction.columns.str.replace('-', '_')
    test_identity.columns = test_identity.columns.str.replace('-', '_')

    # Merge transaction and identity columns
    train_df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
    test_df = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
    print(f"Data loaded and merged. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def process_features(train_df, test_df):
    categorical_features = [
        'ProductCD',
        'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2',
        'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18',
        'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
        'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
        'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
    ]

    # Add one-hot encoding of categorical features
    train_encoded, test_encoded = one_hot_encode_with_threshold(train_df, test_df, categorical_features, threshold=0.01)

    # Normalize all features
    train_scaled, test_scaled = z_scale(train_encoded, test_encoded)

    # Impute missing values
    train_processed, test_processed = impute_missing_values(train_scaled, test_scaled)

    return train_processed, test_processed



def impute_missing_values(train_df, test_df):
    imputer = SimpleImputer(strategy='mean')
    train_imputed = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
    test_imputed = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)

    print("\nMissing value imputation completed.")
    print("  Train dataframe shape:", train_imputed.shape)
    print("  Test dataframe shape:", test_imputed.shape)

    return train_imputed, test_imputed


def one_hot_encode_with_threshold(train_df, test_df, categorical_features, threshold=0.01):
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for feature in categorical_features:
        # Calculate the frequency of each category in the train set
        value_counts = train_df[feature].value_counts(normalize=True)

        # Filter categories that meet the threshold
        valid_categories = value_counts[value_counts >= threshold].index.tolist()

        print(f"Feature '{feature}':")
        print(f"  Valid categories (above threshold): {valid_categories}")
        print(f"  Categories replaced with 'Other': {list(set(value_counts.index) - set(valid_categories))}")

        # Replace categories below the threshold with 'Other' in both train and test
        train_encoded[feature] = train_encoded[feature].apply(lambda x: x if x in valid_categories else 'Other')
        test_encoded[feature] = test_encoded[feature].apply(lambda x: x if x in valid_categories else 'Other')

        # Apply one-hot encoding
        train_one_hot = pd.get_dummies(train_encoded[feature], prefix=feature)
        test_one_hot = pd.get_dummies(test_encoded[feature], prefix=feature)

        # Align train and test one-hot encoded columns
        train_encoded = pd.concat([train_encoded.drop(columns=[feature]), train_one_hot], axis=1)
        test_encoded = pd.concat([test_encoded.drop(columns=[feature]), test_one_hot], axis=1)
        test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    print("\nOne-hot encoding completed.")
    print("  Train dataframe shape:", train_encoded.shape)
    print("  Test dataframe shape:", test_encoded.shape)

    return train_encoded, test_encoded


def z_scale(train_df, test_df, exclude_column='isFraud'):
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # Identify columns to scale
    columns_to_scale = [col for col in train_df.columns if col != exclude_column]

    # Apply StandardScaler to the selected columns
    scaler = StandardScaler()
    train_scaled[columns_to_scale] = scaler.fit_transform(train_scaled[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test_scaled[columns_to_scale])

    print("\nZ-scaling completed.")
    print("  Train dataframe shape:", train_scaled.shape)
    print("  Test dataframe shape:", test_scaled.shape)

    return train_scaled, test_scaled


def split_data(df, test_size=0.2, random_state=42):
    # Separate features (X) and target (y)
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Data split completed:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")

    return X_train, X_val, y_train, y_val


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


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
    results = []
    best_model = None
    best_val_auc = -1  # Initialize with a very low value

    for model in models:
        print(f"Training model: {model.__class__.__name__}")
        trained_model = train_model(model, X_train, y_train)
        result = evaluate_model(trained_model, X_train, X_val, y_train, y_val)
        results.append({
            "model_name": model.__class__.__name__,
            "train_auc": result["train_auc"],
            "val_auc": result["val_auc"]
        })

        # Update the best model based on validation AUC
        if result["val_auc"] > best_val_auc:
            best_val_auc = result["val_auc"]
            best_model = trained_model

    results_df = pd.DataFrame(results)
    print("\nModel Evaluation Results:")
    print(results_df)

    print(f"\nBest Model: {best_model.__class__.__name__}")
    print(f"  Validation AUC: {best_val_auc:.4f}")

    return results_df, best_model


def main_model_evaluation(X_train, X_val, y_train, y_val):
    # Define models
    models = [
        LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
        XGBClassifier(n_estimators = 32, max_depth = 4),
        XGBClassifier(n_estimators = 64, max_depth = 4),
        XGBClassifier(n_estimators = 32, max_depth = 8), # best so far
    ]

    # Train and evaluate models
    results_df, best_model = train_and_evaluate_models(models, X_train, X_val, y_train, y_val)

    # Visualize results of best model
    plot_and_save_roc_curve(best_model, X_val, y_val)
    plot_and_save_confusion_matrix(best_model, X_val, y_val)

    return results_df, best_model


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


def main():
    # Load data
    train_df, test_df = load_data('data/ieee-fraud-detection.zip', n_rows = None) # n_rows = 5000

    # Process features (one-hot encoding of categorical features, z-scaling of all features, imputation of missing values)
    train_df, test_df = process_features(train_df, test_df)

    # Prepare the training and validation sets
    X_train, X_val, y_train, y_val = split_data(train_df)

    # Train and evaluate models
    results_df, best_model = main_model_evaluation(X_train, X_val, y_train, y_val)
    print(results_df)

    # TODO: Add hyperparameters to the model evaluation results df to aid interpretation

    # TODO: Plot feature importance in best model

    # TODO: Train neural network (latter maybe trained on features with >1% importance in XGBoost)

    # TODO: Make predictions for Kaggle competition for the test_df, based on best model



if __name__ == '__main__':
    main()

