import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Use the same seed so data splits and model training are more reproducible
def split_data(X, y, classification=False):
    # Create an 80/10/10 split and preserve class proportions for classification tasks
    stratify = y if classification else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=stratify
    )

    temp_stratify = y_temp if classification else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_stratify
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# Compare Scikit-Learn and Keras neural networks on breast cancer classification

cancer = pd.read_csv("data_refined.csv")

# Separate the input features from the diagnosis target
X = cancer.drop(columns="diagnosis")
y = cancer["diagnosis"]

X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y, classification=True
)

# Train a Scikit-Learn multilayer perceptron with early stopping to help reduce overfitting
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=3000,
    early_stopping=True,
    random_state=SEED
)

mlp_classifier.fit(X_train, y_train)
sk_class_pred = mlp_classifier.predict(X_test)

print("\nSCIKIT-LEARN CLASSIFICATION")
print("Accuracy:", accuracy_score(y_test, sk_class_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, sk_class_pred))


# Scale the features before training the Keras classifier so inputs are on comparable ranges
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

# Use ReLU hidden layers and a sigmoid output for binary classification
keras_classifier = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

keras_classifier.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Monitor validation performance and restore the best weights when training stops
keras_classifier.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=32,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True
        )
    ]
)

# Convert predicted probabilities into class labels using a 0.5 threshold
keras_class_pred = (
    keras_classifier.predict(X_test_scaled, verbose=0).ravel() >= 0.5
).astype(int)

print("\nKERAS CLASSIFICATION")
print("Accuracy:", accuracy_score(y_test, keras_class_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, keras_class_pred))


# Compare Scikit-Learn and Keras neural networks on insurance charge regression

insurance = pd.read_csv("insurance_clustered(2).csv")

# Remove the target and previous cluster labels so they are not used as predictors
X = insurance.drop(
    columns=["charges", "KMeans_Cluster", "MeanShift_Cluster"],
    errors="ignore"
).copy()

y = insurance["charges"]

# Convert Boolean features to numerical values so the models can use them
for column in X.columns:
    if X[column].dtype == bool:
        X[column] = X[column].astype(int)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# Scale both the predictors and target while training the Scikit-Learn neural-network regressor
mlp_regressor = TransformedTargetRegressor(
    regressor=Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=3000,
            early_stopping=True,
            random_state=SEED
        ))
    ]),
    transformer=StandardScaler()
)

mlp_regressor.fit(X_train, y_train)
sk_reg_pred = mlp_regressor.predict(X_test)

print("\nSCIKIT-LEARN REGRESSION")
print("R2 Score:", r2_score(y_test, sk_reg_pred))


# Scale the predictors and target before training the Keras regression network
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).ravel()

y_val_scaled = y_scaler.transform(
    y_val.to_numpy().reshape(-1, 1)
).ravel()

# Use progressively smaller hidden layers and one linear output for regression
keras_regressor = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])

keras_regressor.compile(
    optimizer="adam",
    loss="mse"
)

# Use validation loss for early stopping and keep the best-performing weights
keras_regressor.fit(
    X_train_scaled,
    y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=500,
    batch_size=32,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(
            patience=30,
            restore_best_weights=True
        )
    ]
)

keras_reg_scaled_pred = keras_regressor.predict(
    X_test_scaled, verbose=0
)

keras_reg_pred = y_scaler.inverse_transform(
    keras_reg_scaled_pred
).ravel()

print("\nKERAS REGRESSION")
print("R2 Score:", r2_score(y_test, keras_reg_pred))


# Summarise the outputs so the Scikit-Learn and Keras results can be compared

print("\nFINAL RESULTS")
print("-" * 50)
print(
    "Scikit classification accuracy:",
    accuracy_score(y_test if False else cancer["diagnosis"].iloc[:0], [])
    if False else "See classification output above"
)
print("Compare the two classification accuracies above.")
print("Compare the two regression R2 scores above.")