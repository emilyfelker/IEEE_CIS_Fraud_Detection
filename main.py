import zipfile
import pandas as pd


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

    train_df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
    test_df = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
    print(f"Data loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def main():
    # Load data
    train_df, test_df = load_data('data/ieee-fraud-detection.zip')

    # TODO: Process data and work on features (for both train_df and test_df)

    # TODO: Prepare the training and validation sets from the train_df

    # TODO: Train and evaluate models

    # TODO: Identify the best model based on AUC (Kaggle competition metric)

    # TODO: Make predictions for Kaggle competition for the test_df, based on best model


if __name__ == '__main__':
    main()

