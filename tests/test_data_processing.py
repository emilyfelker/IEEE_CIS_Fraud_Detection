from IEEE_CIS_Fraud_Detection.data_processing import load_data, process_features, split_data
import pandas as pd


def test_split_data():
    # Test data splitting
    df = pd.DataFrame({
        'ProductCD': ['A', 'B', 'A', 'B'],
        'card1': ['123', '456', '789', '012'],
        'isFraud': [0, 1, 0, 1]
    })
    X_train, X_val, y_train, y_val = split_data(df, test_size=0.5)
    assert X_train.shape[0] == 2  # Half of the data
    assert X_val.shape[0] == 2
    assert 'isFraud' not in X_train.columns  # Ensure target is removed from features