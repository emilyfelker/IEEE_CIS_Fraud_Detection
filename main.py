import zipfile
import pandas as pd
from sklearn.preprocessing import StandardScaler


# TODO: Generate ipython notebook for submission to competition
# TODO: Write ReadMe file for GitHub

def load_data(zip_file_path):
    try: # for local environment
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open('train_transaction.csv') as train_transaction:
                train_transaction = pd.read_csv(train_transaction, index_col='TransactionID')
            with z.open('train_identity.csv') as train_identity:
                train_identity = pd.read_csv(train_identity, index_col='TransactionID')
            with z.open('test_transaction.csv') as test_transaction:
                test_transaction = pd.read_csv(test_transaction, index_col='TransactionID')
            with z.open('test_identity.csv') as test_identity:
                test_identity = pd.read_csv(test_identity, index_col='TransactionID')
    except (FileNotFoundError, zipfile.BadZipFile): # for Kaggle environment
        train_transaction = pd.read_csv(
            '/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')
        train_identity = pd.read_csv(
            '/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')
        test_transaction = pd.read_csv(
            '/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')
        test_identity = pd.read_csv(
            '/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')
    print("CSV files read in.")
    train_df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
    test_df = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
    print(f"Data loaded and merged. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def process_features(df):
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
    encoded_df = one_hot_encode_with_threshold(df, categorical_features, threshold=0.01)
    processed_df = z_scale(encoded_df)
    return processed_df


def one_hot_encode_with_threshold(df, categorical_features, threshold=0.01):
    df_encoded = df.copy()

    for feature in categorical_features:
        # Calculate the frequency of each category
        value_counts = df[feature].value_counts(normalize=True)

        # Filter categories that meet the threshold
        valid_categories = value_counts[value_counts >= threshold].index.tolist()

        print(f"Feature '{feature}':")
        print(f"  Valid categories (above threshold): {valid_categories}")
        print(f"  Categories replaced with 'Other': {list(set(value_counts.index) - set(valid_categories))}")

        # Replace categories below the threshold with 'Other'
        df_encoded[feature] = df_encoded[feature].apply(lambda x: x if x in valid_categories else 'Other')

        # Apply one-hot encoding
        one_hot = pd.get_dummies(df_encoded[feature], prefix=feature)
        df_encoded = pd.concat([df_encoded.drop(columns=[feature]), one_hot], axis=1)

    print("\nOne-hot encoding completed. Encoded dataframe shape:", df_encoded.shape)
    return df_encoded


def z_scale(df, exclude_column='isFraud'):
    df_scaled = df.copy()

    # Identify columns to scale
    columns_to_scale = [col for col in df.columns if col != exclude_column]

    # Apply StandardScaler to the selected columns
    scaler = StandardScaler()
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

    print("\nZ-scaling completed. Scaled dataframe shape:", df_scaled.shape)
    return df_scaled


def main():
    # Load data
    train_df, test_df = load_data('data/ieee-fraud-detection.zip')

    # Process features (one-hot encoding of categorical features, then z-scaling of all features)
    train_df = process_features(train_df)
    test_df = process_features(test_df)

    # TODO: Prepare the training and validation sets from the train_df

    # TODO: Train and evaluate models
    # Start with logistic regression, XGboost, k-nearest-neighbor
    # Could use xgboost to select all features with importance > 1% & train neural network on those ones

    # TODO: Identify the best model based on AUC (Kaggle competition metric)

    # TODO: Make predictions for Kaggle competition for the test_df, based on best model



if __name__ == '__main__':
    main()

