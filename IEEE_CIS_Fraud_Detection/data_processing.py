import zipfile
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Optional


def load_data(zip_file_path: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open('train_transaction.csv') as train_transaction:
            train_transaction = pd.read_csv(train_transaction, index_col='TransactionID', nrows=n_rows)
        with z.open('train_identity.csv') as train_identity:
            train_identity = pd.read_csv(train_identity, index_col='TransactionID', nrows=n_rows)
    print("CSV files read in.")
    df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
    print(f"Data loaded and merged. Shape: {df.shape}")

    return df


def process_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df_encoded = one_hot_encode_with_threshold(df, categorical_features, threshold=0.01)

    # Impute missing values
    df_imputed = impute_missing_values(df_encoded)

    # Standardize all features
    df_scaled = z_scale(df_imputed)

    return df_scaled


def impute_missing_values(df: pd.DataFrame, exclude_column: str = 'isFraud') -> pd.DataFrame:
    df_imputed = df.copy()

    # Identify columns to impute
    columns_to_impute = [col for col in df.columns if col != exclude_column]

    imputer = SimpleImputer(strategy='mean')
    df_imputed[columns_to_impute] = imputer.fit_transform(df_imputed[columns_to_impute])

    print(f"Missing value imputation completed. Dataframe shape: {df_imputed.shape}")

    return df_imputed


def one_hot_encode_with_threshold(df: pd.DataFrame, categorical_features: list[str],
                                  threshold: float = 0.01) -> pd.DataFrame:
    encoder = OneHotEncoder(
        min_frequency=threshold,
        handle_unknown="infrequent_if_exist",  # Combine rare categories into 'infrequent'
        sparse_output=False,  # Return dense arrays
    )

    df_encoded_array = encoder.fit_transform(df[categorical_features])

    # Extract column names from encoder
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)

    # Convert to DataFrame
    df_encoded = pd.DataFrame(df_encoded_array, columns=encoded_feature_names, index=df.index)

    # Drop original categorical features and append encoded features
    df_encoded = pd.concat([df.drop(columns=categorical_features), df_encoded], axis=1)

    print(f"One-hot encoding with threshold completed. Dataframe shape: {df_encoded.shape}")

    return df_encoded


def z_scale(df: pd.DataFrame, exclude_column='isFraud') -> pd.DataFrame:
    df_scaled = df.copy()

    # Identify columns to scale
    columns_to_scale = [col for col in df.columns if col != exclude_column]

    # Apply StandardScaler to the selected columns
    scaler = StandardScaler()
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

    print(f"Z-scaling completed. Dataframe shape: {df_scaled.shape}")

    return df_scaled


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> \
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Separate features (X) and target (y)
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Data split completed:")
    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")

    return X_train, X_val, y_train, y_val


def reduce_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    # Ensure all specified features exist in the DataFrame
    missing_features = [feature for feature in feature_names if feature not in df.columns]
    if missing_features:
        raise ValueError(f"The following features are missing in the DataFrame: {missing_features}")

    # Add "isFraud" to the feature list if it exists in the DataFrame
    if "isFraud" in df.columns and "isFraud" not in feature_names:
        feature_names = feature_names + ["isFraud"]

    # Select only the specified features
    reduced_df = df[feature_names].copy()

    return reduced_df