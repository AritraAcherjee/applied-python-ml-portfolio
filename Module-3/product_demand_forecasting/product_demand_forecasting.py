import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load the historical demand data from the project CSV
data = pd.read_csv("Historical Product Demand.csv")


# Inspect the original data before any cleaning or transformation
print("\n================ ORIGINAL DATASET ================")
print(data.head())
print("\nDataset shape:")
print(data.shape)
print("\nColumn names:")
print(data.columns.tolist())
print("\nDataset information:")
data.info()
print("\nMissing values in original dataset:")
print(data.isnull().sum())


# Remove incomplete rows so later conversions and models receive usable values
data = data.dropna()

# Demand values may contain commas or accounting-style parentheses, so normalize them before conversion
data["Order_Demand"] = (
    data["Order_Demand"]
    .astype(str)
    .str.strip()
    .str.replace(",", "", regex=False)
    .str.replace("(", "-", regex=False)
    .str.replace(")", "", regex=False)
)


# Any remaining invalid demand value becomes missing rather than causing a conversion error
data["Order_Demand"] = pd.to_numeric(
    data["Order_Demand"],
    errors="coerce"
)


# Convert the date column so useful calendar features can be extracted
data["Date"] = pd.to_datetime(
    data["Date"]
)


# Calendar features let the models capture seasonal and time-related demand patterns
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["DayOfWeek"] = data["Date"].dt.dayofweek
data["Quarter"] = data["Date"].dt.quarter


# The original date is no longer needed after its components have been extracted
data = data.drop(columns=["Date"])

# Remove duplicate records so repeated rows do not receive extra influence during training
data = data.drop_duplicates()


print("\n================ CLEANED DATASET ================")

print(data.head())


print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nCleaned dataset shape:")
print(data.shape)

# Separate predictor columns from the demand value the models will learn to estimate
X = data.drop(
    columns=["Order_Demand"]
)

y = data["Order_Demand"]


# Categorical and numerical columns need different preprocessing before modeling
categorical_features = X.select_dtypes(
    include=[
        "object",
        "category"
    ]
).columns.tolist()


numerical_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)


print("\nNumerical features:")
print(numerical_features)


# First reserve 20% of the data, leaving 80% for model training
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Split the reserved data equally so validation can guide model choice while test data stays unseen
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)


print("\n================ DATA SPLIT ================")

print(
    "Training rows:",
    len(X_train)
)

print(
    "Validation rows:",
    len(X_validation)
)

print(
    "Testing rows:",
    len(X_test)
)


# Scaling keeps numerical features on a comparable range, which is especially important for SVR
numerical_transformer = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Convert categories to numerical indicators and ignore unseen categories at prediction time
categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


# Apply the appropriate transformation to each feature type in one reusable preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_transformer,
            numerical_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# Report complementary regression metrics: fit quality, average error, and error with larger misses emphasized
def print_results(
    model_name,
    actual_values,
    predicted_values
):

    r2 = r2_score(
        actual_values,
        predicted_values
    )


    mae = mean_absolute_error(
        actual_values,
        predicted_values
    )


    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values
        )
    )


    print(
        f"\n{model_name}"
    )

    print("-" * 55)


    print(
        f"R-squared score:          {r2:.4f}"
    )

    print(
        f"Mean Absolute Error:      {mae:.2f}"
    )

    print(
        f"Root Mean Squared Error:  {rmse:.2f}"
    )


print("\n")
print("=" * 65)
print("DECISION TREE REGRESSION")
print("=" * 65)


# Compare several Decision Tree criteria on validation data instead of choosing with the final test set
tree_criteria = [
    "squared_error",
    "friedman_mse",
    "absolute_error"
]


best_tree_model = None
best_tree_criterion = None
best_tree_validation_r2 = float("-inf")


# Train the same pipeline with each criterion so preprocessing stays consistent across comparisons
for criterion in tree_criteria:

    decision_tree_pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                DecisionTreeRegressor(
                    criterion=criterion,
                    random_state=42
                )
            )
        ]
    )


    decision_tree_pipeline.fit(
        X_train,
        y_train
    )


    tree_validation_predictions = (
        decision_tree_pipeline.predict(
            X_validation
        )
    )


    validation_r2 = r2_score(
        y_validation,
        tree_validation_predictions
    )


    print(
        f"Decision Tree criterion={criterion} "
        f"| Validation R² = {validation_r2:.4f}"
    )


    if validation_r2 > best_tree_validation_r2:

        best_tree_validation_r2 = validation_r2

        best_tree_criterion = criterion

        best_tree_model = decision_tree_pipeline


print(
    "\nBest Decision Tree criterion:",
    best_tree_criterion
)


print(
    "Best Decision Tree validation R²:",
    round(
        best_tree_validation_r2,
        4
    )
)


# Evaluate the selected tree on training, validation, and finally unseen test data
tree_train_predictions = best_tree_model.predict(
    X_train
)


tree_validation_predictions = best_tree_model.predict(
    X_validation
)


tree_test_predictions = best_tree_model.predict(
    X_test
)


print_results(
    "Decision Tree - Training Results",
    y_train,
    tree_train_predictions
)


print_results(
    "Decision Tree - Validation Results",
    y_validation,
    tree_validation_predictions
)


print_results(
    "Decision Tree - Final Testing Results",
    y_test,
    tree_test_predictions
)


print("\n")
print("=" * 65)
print("SUPPORT VECTOR REGRESSION")
print("=" * 65)


# RBF SVR can model nonlinear relationships between the processed features and product demand
svr_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            SVR(
                kernel="rbf",
                C=100,
                epsilon=0.1,
                gamma="scale"
            )
        )
    ]
)


print(
    "\nTraining the SVR model. "
    "This dataset is very large, so this may take a long time..."
)


svr_pipeline.fit(
    X_train,
    y_train
)

svr_train_predictions = svr_pipeline.predict(
    X_train
)

svr_validation_predictions = svr_pipeline.predict(
    X_validation
)

svr_test_predictions = svr_pipeline.predict(
    X_test
)

print_results(
    "SVR - Training Results",
    y_train,
    svr_train_predictions
)

print_results(
    "SVR - Validation Results",
    y_validation,
    svr_validation_predictions
)

print_results(
    "SVR - Final Testing Results",
    y_test,
    svr_test_predictions
)


# Use the held-out test R-squared values for the final model comparison
decision_tree_test_r2 = r2_score(
    y_test,
    tree_test_predictions
)


svr_test_r2 = r2_score(
    y_test,
    svr_test_predictions
)

print("\n")
print("=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)


print(
    f"Decision Tree Regression test R²: "
    f"{decision_tree_test_r2:.4f}"
)


print(
    f"Support Vector Regression test R²: "
    f"{svr_test_r2:.4f}"
)


if decision_tree_test_r2 > svr_test_r2:

    print(
        "\nThe Decision Tree produced "
        "the higher test R² score."
    )


elif svr_test_r2 > decision_tree_test_r2:

    print(
        "\nThe SVR model produced "
        "the higher test R² score."
    )


else:

    print(
        "\nBoth models produced "
        "the same test R² score."
    )


print(
    "\nProject completed successfully."
)