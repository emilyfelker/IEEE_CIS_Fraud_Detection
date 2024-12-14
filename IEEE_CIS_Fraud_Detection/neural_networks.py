import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
from sklearn.metrics import roc_auc_score


def build_neural_network(input_dim, gelu_layer_sizes:list[int]=[64], last_layer:int=16):
    gelu_layers = []
    for n in gelu_layer_sizes:
        gelu_layers.append(Dense(n, activation='gelu'))
        gelu_layers.append(Dropout(0.5))

    model = Sequential([
        Input(shape=(input_dim,)),
        *gelu_layers,
        Dense(last_layer, activation='gelu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['AUC'])

    return model


def train_neural_networks(X_train, X_val, y_train, y_val):
    input_dim = X_train.shape[1]
    models = [
        build_neural_network(input_dim, gelu_layer_sizes=[32], last_layer=16),
        build_neural_network(input_dim, gelu_layer_sizes=[64], last_layer=16),
        build_neural_network(input_dim, gelu_layer_sizes=[128], last_layer=16),
        build_neural_network(input_dim, gelu_layer_sizes=[64, 32], last_layer=16),
        build_neural_network(input_dim, gelu_layer_sizes=[32, 16], last_layer=8),
        build_neural_network(input_dim, gelu_layer_sizes=[16], last_layer=8),
    ]

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_AUC', patience=5, restore_best_weights=True, mode='max')

    # Handling imbalanced data with weighted loss function
    class_weights = {0: 1.0, 1: 10.0}
    y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train # doesn't work without this

    trained_models = []
    for model in models:
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=256,
            epochs=50,
            callbacks=[early_stopping],
            verbose=1,
            class_weight=class_weights
        )

        # Evaluate the model
        val_probs = model.predict(X_val).flatten()
        train_probs = model.predict(X_train).flatten()

        train_auc = roc_auc_score(y_train, train_probs)
        val_auc = roc_auc_score(y_val, val_probs)

        print(f"Neural Network:")
        print(model.summary())
        print(f"  Training AUC: {train_auc:.4f}")
        print(f"  Validation AUC: {val_auc:.4f}")
        trained_models.append((model, train_auc, val_auc))

    return trained_models