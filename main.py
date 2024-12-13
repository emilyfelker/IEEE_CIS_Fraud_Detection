from data_processing import load_data, process_features, reduce_features, split_data
from data_visualization import plot_and_save_confusion_matrix, plot_and_save_feature_importance, plot_and_save_roc_curve
from training_evaluation import classical_model_evaluation, get_top_features
from neural_networks import train_neural_networks

def main():
    # Load data
    dataset = load_data('data/ieee-fraud-detection.zip', n_rows = None) # n_rows = 5000

    # Process features (one-hot encoding of categorical features, z-scaling of all features, imputation of missing values)
    dataset_processed = process_features(dataset)

    # Prepare the training and validation sets
    X_train, X_val, y_train, y_val = split_data(dataset_processed)

    # Pass feature names for plotting
    feature_names = X_train.columns.tolist()

    # Train and evaluate models (regression and random forest)
    best_model = classical_model_evaluation(X_train, X_val, y_train, y_val, feature_names)

    # Reduce features in order to then train neural network
    top_features = get_top_features(best_model, feature_names, n=512)
    df_reduced = reduce_features(dataset_processed, top_features)
    X_train_r, X_val_r, y_train_r, y_val_r = split_data(df_reduced)

    # Train neural network on reduced dataset
    trained_nn_models = train_neural_networks(X_train_r, X_val_r, y_train_r, y_val_r)

    # Visualize results of best model
    plot_and_save_roc_curve(best_model, X_val, y_val)
    plot_and_save_confusion_matrix(best_model, X_val, y_val)
    plot_and_save_feature_importance(best_model, feature_names)


if __name__ == '__main__':
    main()