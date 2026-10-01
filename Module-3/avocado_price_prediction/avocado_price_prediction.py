import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



# Load the avocado dataset directly from the project CSV
df = pd.read_csv("avocado.csv")

print("First five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())


print("\nMissing values before cleaning:")
print(df.isnull().sum())


# Remove incomplete rows so the regression models receive usable feature values
df = df.dropna()

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# Region and Date are excluded from this version of the prediction model
columns_to_remove = ["region", "Date"]

for column in columns_to_remove:
    if column in df.columns:
        df = df.drop(columns=column)


# Separate the predictor columns from AveragePrice, which is the regression target
X = df.drop(columns=["AveragePrice"])
y = df["AveragePrice"]

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget column:")
print(y.name)


# Numerical and categorical columns need different preprocessing before modeling
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)



# Reserve 20% of the data while keeping 80% for initial model training
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Split the reserved data equally so validation guides model choice and test data stays unseen
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("\nDataset sizes:")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Testing rows:", len(X_test))


# Scale numerical values for distance-based KNN and encode categories for both regression models
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            # Ignore categories that may appear later but were not present during fitting
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)



# Compare several neighborhood sizes on validation data instead of choosing k from test performance
k_values = [3, 5, 7, 9, 11, 13, 15]

best_k = None
best_validation_score = float("-inf")

print("\nValidation results:")

for k in k_values:


    knn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=k
                )
            )
        ]
    )


    knn_pipeline.fit(X_train, y_train)


    validation_predictions = knn_pipeline.predict(X_validation)


    validation_score = r2_score(
        y_validation,
        validation_predictions
    )

    print(f"k = {k}: Validation R-squared = {validation_score:.4f}")


    # Keep the k value with the strongest validation R-squared
    if validation_score > best_validation_score:
        best_validation_score = validation_score
        best_k = k


print("\nBest value of k:", best_k)
print(f"Best validation R-squared: {best_validation_score:.4f}")


# After selecting k, combine training and validation data so the final model learns from more observations
X_final_train = pd.concat(
    [X_train, X_validation],
    axis=0
)

y_final_train = pd.concat(
    [y_train, y_validation],
    axis=0
)


final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=best_k
            )
        )
    ]
)


final_model.fit(X_final_train, y_final_train)


# Evaluate the selected KNN model only after k has been fixed using validation data
test_predictions = final_model.predict(X_test)

test_r2_score = r2_score(
    y_test,
    test_predictions
)

print("\nFinal KNN regression results:")
print("Best k:", best_k)
print(f"Test R-squared score: {test_r2_score:.4f}")



# Train Linear Regression with the same preprocessing for a fair comparison with KNN
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "linear_regression",
            LinearRegression()
        )
    ]
)

linear_model.fit(X_final_train, y_final_train)

linear_predictions = linear_model.predict(X_test)

linear_r2_score = r2_score(
    y_test,
    linear_predictions
)

print("\nLinear Regression results:")
print(f"Test R-squared score: {linear_r2_score:.4f}")

print("\nModel comparison:")
print(f"KNN R-squared: {test_r2_score:.4f}")
print(f"Linear Regression R-squared: {linear_r2_score:.4f}")

# Compare both models using the same held-out test set
if test_r2_score > linear_r2_score:
    print("KNN Regression performed better.")

elif linear_r2_score > test_r2_score:
    print("Linear Regression performed better.")

else:
    print("Both models had the same R-squared score.")

# Show a small sample of actual prices beside predictions from both models
results = pd.DataFrame(
    {
        "Actual Price": y_test.values,
        "KNN Predicted Price": test_predictions,
        "Linear Regression Predicted Price": linear_predictions
    }
)

print("\nSample actual and predicted avocado prices:")
print(results.head(10))