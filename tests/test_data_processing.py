from IEEE_CIS_Fraud_Detection.data_processing import load_data, process_features, split_data
import pandas as pd
import numpy as np
from unittest import mock
from io import StringIO


def test_load_data():
    # Arrange
    mock_train_transaction = pd.DataFrame({
        'TransactionID': [1, 2],
        'feature1': ['A', 'B'],
        'feature2': [10, 20]
    })
    mock_train_identity = pd.DataFrame({
        'TransactionID': [1, 2],
        'feature3': ['X', 'Y'],
        'feature4': [100, 200]
    })

    with mock.patch('zipfile.ZipFile') as MockZipFile:
        mock_zip_instance = MockZipFile.return_value.__enter__.return_value

        # Mock separate file objects for each `z.open` call
        mock_file_transaction = StringIO("mocked CSV content")
        mock_file_identity = StringIO("mocked CSV content")
        mock_zip_instance.open.side_effect = [mock_file_transaction, mock_file_identity]

        # Mock pandas.read_csv to return mock DataFrames
        with mock.patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [mock_train_transaction, mock_train_identity]

            # Act
            df = load_data('dummy_path.zip', n_rows=2)

    # Assert
    # After merging, we expect 6 columns (with duplicated 'TransactionID')
    assert df.shape == (2, 6)

    # Check that pandas.read_csv was called for both files
    mock_read_csv.assert_any_call(mock_file_transaction, index_col='TransactionID', nrows=2)
    mock_read_csv.assert_any_call(mock_file_identity, index_col='TransactionID', nrows=2)

    # Verify that the merged DataFrame has the expected columns
    expected_columns = ['TransactionID_x', 'feature1', 'feature2', 'feature3', 'feature4', 'TransactionID_y']
    assert all(col in df.columns for col in expected_columns), df.columns

    assert df.loc[df['TransactionID_x'] == 1, 'feature3'].iloc[0] == 'X'
    assert df.loc[df['TransactionID_x'] == 2, 'feature3'].iloc[0] == 'Y'
    assert df.loc[df['TransactionID_x'] == 1, 'feature4'].iloc[0] == 100
    assert df.loc[df['TransactionID_x'] == 2, 'feature4'].iloc[0] == 200


def test_process_features():
    # Arrange: Create a dataframe with all categorical features, continuous features, 'isFraud', and some missing values
    data = {
        # Categorical features
        'ProductCD': ['A', 'B', 'A', 'C'],
        'card1': ['123', '456', '123', '789'],
        'card2': ['001', '002', '001', '003'],
        'card3': ['AA', 'BB', 'AA', 'CC'],
        'card4': ['X', 'Y', 'X', 'Z'],
        'card5': ['1', '2', '1', '3'],
        'card6': ['M', 'N', 'M', 'O'],
        'addr1': ['100', '200', '100', '300'],
        'addr2': ['50', '60', '50', '70'],
        'P_emaildomain': ['gmail.com', 'yahoo.com', 'gmail.com', 'hotmail.com'],
        'R_emaildomain': ['gmail.com', 'yahoo.com', 'hotmail.com', 'gmail.com'],
        'M1': ['T', 'F', 'T', 'F'],
        'M2': ['T', 'F', 'F', 'T'],
        'M3': ['T', 'F', 'T', 'F'],
        'M4': ['T', 'F', 'T', 'F'],
        'M5': ['T', 'F', 'T', 'F'],
        'M6': ['T', 'F', 'T', 'F'],
        'M7': ['T', 'F', 'T', 'F'],
        'M8': ['T', 'F', 'T', 'F'],
        'M9': ['T', 'F', 'T', 'F'],
        'DeviceType': ['mobile', 'desktop', 'mobile', 'tablet'],
        'DeviceInfo': ['info1', 'info2', 'info1', 'info3'],
        'id_12': ['a', 'b', 'a', 'c'],
        'id_13': ['d', 'e', 'd', 'f'],
        'id_14': ['g', 'h', 'g', 'i'],
        'id_15': ['j', 'k', 'j', 'l'],
        'id_16': ['m', 'n', 'm', 'o'],
        'id_17': ['p', 'q', 'p', 'r'],
        'id_18': ['s', 't', 's', 'u'],
        'id_19': ['v', 'w', 'v', 'x'],
        'id_20': ['y', 'z', 'y', 'a1'],
        'id_21': ['b1', 'c1', 'b1', 'd1'],
        'id_22': ['e1', 'f1', 'e1', 'g1'],
        'id_23': ['h1', 'i1', 'h1', 'j1'],
        'id_24': ['k1', 'l1', 'k1', 'm1'],
        'id_25': ['n1', 'o1', 'n1', 'p1'],
        'id_26': ['q1', 'r1', 'q1', 's1'],
        'id_27': ['t1', 'u1', 't1', 'v1'],
        'id_28': ['w1', 'x1', 'w1', 'y1'],
        'id_29': ['z1', 'a2', 'z1', 'b2'],
        'id_30': ['c2', 'd2', 'c2', 'e2'],
        'id_31': ['f2', 'g2', 'f2', 'h2'],
        'id_32': ['i2', 'j2', 'i2', 'k2'],
        'id_33': ['l2', 'm2', 'l2', 'n2'],
        'id_34': ['o2', 'p2', 'o2', 'q2'],
        'id_35': ['r2', 's2', 'r2', 't2'],
        'id_36': ['u2', 'v2', 'u2', 'w2'],
        'id_37': ['x2', 'y2', 'x2', 'z2'],
        'id_38': ['a3', 'b3', 'a3', 'c3'],
        # Continuous features
        'feature1': [1.0, 2.0, 3.0, None],  # Missing value in feature1
        'feature2': [10, 20, 30, 40],
        'feature3': [100, 200, None, 400],
        'feature4': [5.5, None, 7.5, 8.5],
        'feature5': [None, 1.0, 2.0, 3.0],
        # Target variable
        'isFraud': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Act: Process the features
    df_processed = process_features(df)

    # Assert:
    # 1. One-hot encoding was applied to all categorical features
    # (since listing all expected encoded columns is cumbersome, check a representative sample)
    sample_encoded_columns = [
        'ProductCD_A', 'ProductCD_B', 'ProductCD_C',
        'card1_123', 'card1_456', 'card1_789',
        'addr1_100', 'addr1_200', 'addr1_300',
        'P_emaildomain_gmail.com', 'P_emaildomain_yahoo.com', 'P_emaildomain_hotmail.com',
        'DeviceType_mobile', 'DeviceType_desktop', 'DeviceType_tablet',
        'id_12_a', 'id_12_b', 'id_12_c',
        'id_20_a1', 'id_20_y', 'id_20_z'
    ]

    for col in sample_encoded_columns:
        assert col in df_processed.columns, f"Missing expected encoded column: {col}"

    # 2. Check that 'isFraud' column is still present and unchanged in values and type
    assert 'isFraud' in df_processed.columns, "'isFraud' column is missing after processing."
    pd.testing.assert_series_equal(
        df_processed['isFraud'],
        df['isFraud'],
        check_names=False,
        check_dtype=True  # Ensures data type remains unchanged (int64)
    )

    # 3. Check that all features except 'isFraud' are scaled
    # Get all columns except 'isFraud'
    features = df_processed.drop(columns=['isFraud'])

    # Calculate means and stds with ddof=0 to match StandardScaler's behavior
    means = features.mean()
    stds = features.std(ddof=0)

    # Check that means are approximately 0 and stds are approximately 1
    # Allowing a small tolerance due to floating point precision
    for col in features.columns:
        assert abs(means[col]) < 1e-6, f"Feature '{col}' mean is not approximately 0 (mean={means[col]})."
        assert abs(stds[col] - 1) < 1e-6, f"Feature '{col}' std is not approximately 1 (std={stds[col]})."

    # 4. Check that there are no missing values after processing
    assert not df_processed.isnull().values.any(), "There are still missing values after processing."

    # 5. Check that the number of columns has increased due to one-hot encoding
    original_num_columns = len(df.columns)
    processed_num_columns = len(df_processed.columns)
    assert processed_num_columns > original_num_columns, (
        f"Number of columns did not increase after processing. "
        f"Original: {original_num_columns}, Processed: {processed_num_columns}."
    )

    # 6. Verify specific scaled values for feature1 (just as an example)
    # Since feature1 was [1.0, 2.0, 3.0, imputed to 2.0], scaled to approximately [-1.4142, 0.0, 1.4142, 0.0]
    expected_feature1_scaled = [-1.4142, 0.0, 1.4142, 0.0]
    actual_feature1_scaled = df_processed['feature1'].values
    np.testing.assert_allclose(
        actual_feature1_scaled,
        expected_feature1_scaled,
        rtol=1e-4,
        err_msg="Scaled values for feature1 do not match expected values."
    )


def test_split_data():
    # Arrange: Create a DataFrame with sample data
    df = pd.DataFrame({
        'ProductCD': ['A', 'B', 'A', 'B'],
        'card1': ['123', '456', '789', '012'],
        'isFraud': [0, 1, 0, 1]
    })

    # Act: Split the data into training and validation sets (50% each)
    X_train, X_val, y_train, y_val = split_data(df, test_size=0.5)

    # Assert: Check the resulting splits
    assert X_train.shape[0] == 2  # Half of the data
    assert X_val.shape[0] == 2
    assert 'isFraud' not in X_train.columns  # Ensure target is removed from features