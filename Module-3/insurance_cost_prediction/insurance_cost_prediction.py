
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

sns.set_theme(style="whitegrid")

RANDOM_STATE = 42  # Keeps data splits and model results reproducible

# Load the dataset directly from the CSV file
df = pd.read_csv("insurance.csv")

print("\n" + "=" * 70)
print("DATASET LOADED")
print("=" * 70)

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# Take a first look at the dataset
print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

df.info()

print("\nColumn names:")
print(df.columns.tolist())

print("\nDescriptive statistics:")
print(df.describe(include="all"))

# Check for repeated rows before modelling
print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicates = df.duplicated().sum()

print("Number of duplicated rows:", duplicates)

# Only change the dataframe when duplicates are actually present
if duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)

    print("Duplicates removed.")
    print("New dataset shape:", df.shape)

else:
    print("No duplicate rows found.")

# Missing data check
print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_table = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Percentage": (
        df.isnull().sum() / len(df)
    ) * 100
})

print(missing_table)

print(
    "\nTotal missing values:",
    df.isnull().sum().sum()
)

# Look at the range of values in each column
print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in df.columns:

    print(f"\nColumn: {column}")

    if df[column].nunique() <= 20:
        print(df[column].value_counts(dropna=False))

    else:
        print(
            "Number of unique values:",
            df[column].nunique()
        )

# Visual check of relationships between numeric features
print("\nCreating pair plot...")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

sns.pairplot(
    df[numeric_columns],
    diag_kind="hist"
)

plt.show()

# Compare numeric relationships for smokers and non-smokers
if "smoker" in df.columns:

    sns.pairplot(
        df,
        vars=[
            "age",
            "bmi",
            "children",
            "charges"
        ],
        hue="smoker",
        diag_kind="hist"
    )

    plt.show()

# Correlation between numerical variables
correlation_matrix = df.select_dtypes(
    include=np.number
).corr()

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# Use box plots to spot possible outliers
numeric_features_for_plots = [
    col
    for col in [
        "age",
        "bmi",
        "children",
        "charges"
    ]
    if col in df.columns
]

for column in numeric_features_for_plots:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df[column]
    )

    plt.title(
        f"Box Plot of {column}"
    )

    plt.xlabel(column)

    plt.tight_layout()
    plt.show()

# Smoking status is expected to have a strong effect on charges
if "smoker" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="smoker",
        y="charges"
    )

    plt.title(
        "Insurance Charges by Smoker Status"
    )

    plt.xlabel("Smoker")
    plt.ylabel("Charges")

    plt.tight_layout()
    plt.show()

# Compare charges across sex categories
if "sex" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="sex",
        y="charges"
    )

    plt.title(
        "Insurance Charges by Sex"
    )

    plt.tight_layout()
    plt.show()

# Compare regional differences in charges
if "region" in df.columns:

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="region",
        y="charges"
    )

    plt.title(
        "Insurance Charges by Region"
    )

    plt.tight_layout()
    plt.show()

# Distribution of the value the models will predict
plt.figure(figsize=(10, 5))

sns.histplot(
    df["charges"],
    bins=40,
    kde=True
)

plt.title(
    "Distribution of Insurance Charges"
)

plt.xlabel("Charges")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Add a few features that may help the models capture interactions
# BMI 30 is used as the obesity threshold
df["obese"] = (
    df["bmi"] >= 30
).astype(int)

# Convert yes/no values so they can be used in interaction features
df["smoker_numeric"] = (
    df["smoker"]
    .astype(str)
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)

# Capture the combined effect of smoking and obesity
df["smoker_obese"] = (
    df["smoker_numeric"] *
    df["obese"]
)

# Let the model account for BMI differently when smoking is present
df["bmi_smoker_interaction"] = (
    df["bmi"] *
    df["smoker_numeric"]
)

print("\nFeature engineering completed.")

print(df.head())

# Separate predictors from the target variable
TARGET = "charges"

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

print("\n" + "=" * 70)
print("FEATURES AND TARGET")
print("=" * 70)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Keep numeric and categorical columns separate for preprocessing
numerical_features = (
    X.select_dtypes(
        include=np.number
    ).columns.tolist()
)

categorical_features = (
    X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()
)

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

# Preprocessing is handled inside the model pipeline
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            # Median is less affected by extreme numerical values
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            # Scaling keeps numerical features on a comparable range, which is important for SVR
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            # Use the most common category instead of creating a new category for missing values
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            # Ignore categories that may appear later but were not in training
            # Dropping the first level avoids an unnecessary duplicate dummy column
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

print(
    "\nPreprocessing pipeline created successfully."
)

# Split the data into 80% training, 10% validation and 10% testing
# Validation is used to compare models while the test set stays unseen until final evaluation
X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )
)

X_val, X_test, y_val, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE
    )
)

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Test samples:",
    len(X_test)
)

print(
    "\nTraining percentage:",
    round(
        len(X_train) / len(df) * 100,
        2
    ),
    "%"
)

print(
    "Validation percentage:",
    round(
        len(X_val) / len(df) * 100,
        2
    ),
    "%"
)

print(
    "Test percentage:",
    round(
        len(X_test) / len(df) * 100,
        2
    ),
    "%"
)

# Reuse the same evaluation metrics for every model
def evaluate_model(
    model,
    X_data,
    y_true,
    model_name
):

    y_pred = model.predict(
        X_data
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        f"R² Score: {r2:.4f}"
    )

    print(
        f"R² Percentage: {r2 * 100:.2f}%"
    )

    print(
        f"MSE: {mse:,.2f}"
    )

    print(
        f"RMSE: {rmse:,.2f}"
    )

    print(
        f"MAE: {mae:,.2f}"
    )

    return {
        "Model": model_name,
        "R2": r2,
        "R2 (%)": r2 * 100,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    }

# Decision Tree
decision_tree_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            DecisionTreeRegressor(
                random_state=RANDOM_STATE
            )
        )
    ]
)

decision_tree_parameters = {

    "model__criterion": [
        "squared_error",
        "friedman_mse",
        "absolute_error",
        "poisson"
    ],

    "model__max_depth": [
        None,
        3,
        4,
        5,
        6,
        8,
        10
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4,
        6
    ]
}

print("\n" + "=" * 70)
print("TRAINING DECISION TREE")
print("=" * 70)

# Five-fold CV gives each parameter combination several training/validation checks
# R² is used because the assessment target is based on an R² score
dt_grid_search = GridSearchCV(
    estimator=decision_tree_pipeline,
    param_grid=decision_tree_parameters,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1
)

dt_grid_search.fit(
    X_train,
    y_train
)

best_decision_tree = (
    dt_grid_search.best_estimator_
)

print(
    "\nBest Decision Tree Parameters:"
)

print(
    dt_grid_search.best_params_
)

print(
    "\nBest Decision Tree CV R²:"
)

print(
    round(
        dt_grid_search.best_score_,
        4
    )
)

# Check Decision Tree performance on validation data
dt_validation_results = evaluate_model(
    best_decision_tree,
    X_val,
    y_val,
    "Decision Tree - Validation"
)

# Random Forest
random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

random_forest_parameters = {

    "model__criterion": [
        "squared_error",
        "friedman_mse",
        "absolute_error"
    ],

    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        None,
        5,
        8,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ]
}

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

# Use the same five-fold approach so the Random Forest is compared consistently
rf_grid_search = GridSearchCV(
    estimator=random_forest_pipeline,
    param_grid=random_forest_parameters,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1
)

rf_grid_search.fit(
    X_train,
    y_train
)

best_random_forest = (
    rf_grid_search.best_estimator_
)

print(
    "\nBest Random Forest Parameters:"
)

print(
    rf_grid_search.best_params_
)

print(
    "\nBest Random Forest CV R²:"
)

print(
    round(
        rf_grid_search.best_score_,
        4
    )
)

# Validate the tuned Random Forest
rf_validation_results = evaluate_model(
    best_random_forest,
    X_val,
    y_val,
    "Random Forest - Validation"
)

# Support Vector Regression
svr_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            SVR()
        )
    ]
)

svr_parameters = {

    "model__kernel": [
        "rbf"
    ],

    "model__C": [
        100,
        1000,
        5000,
        10000
    ],

    "model__gamma": [
        "scale",
        0.01,
        0.05,
        0.1
    ],

    "model__epsilon": [
        0.1,
        100,
        500,
        1000
    ]
}

print("\n" + "=" * 70)
print("TRAINING SUPPORT VECTOR REGRESSION")
print("=" * 70)

# Search several SVR settings instead of relying on one default configuration
# SVR can be sensitive to C, gamma and epsilon, so these are tuned rather than assumed
svr_grid_search = GridSearchCV(
    estimator=svr_pipeline,
    param_grid=svr_parameters,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1
)

svr_grid_search.fit(
    X_train,
    y_train
)

best_svr = (
    svr_grid_search.best_estimator_
)

print(
    "\nBest SVR Parameters:"
)

print(
    svr_grid_search.best_params_
)

print(
    "\nBest SVR CV R²:"
)

print(
    round(
        svr_grid_search.best_score_,
        4
    )
)

# Validate the tuned SVR model
svr_validation_results = evaluate_model(
    best_svr,
    X_val,
    y_val,
    "SVR - Validation"
)

# Compare the three models using validation results
validation_results = pd.DataFrame(
    [
        dt_validation_results,
        rf_validation_results,
        svr_validation_results
    ]
)

validation_results = (
    validation_results
    .sort_values(
        by="R2",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(validation_results)

# Plot validation R² so the model differences are easy to see
plt.figure(
    figsize=(9, 5)
)

sns.barplot(
    data=validation_results,
    x="Model",
    y="R2"
)

plt.axhline(
    y=0.82,
    linestyle="--",
    label="Required R² = 82%"
)

plt.title(
    "Validation R² Comparison"
)

plt.ylabel(
    "R² Score"
)

plt.xlabel(
    "Model"
)

plt.xticks(
    rotation=15
)

plt.legend()

plt.tight_layout()
plt.show()

# Choose the final model using validation R² only
models = {
    "Decision Tree": best_decision_tree,
    "Random Forest": best_random_forest,
    "SVR": best_svr
}

validation_scores = {

    "Decision Tree":
        r2_score(
            y_val,
            best_decision_tree.predict(
                X_val
            )
        ),

    "Random Forest":
        r2_score(
            y_val,
            best_random_forest.predict(
                X_val
            )
        ),

    "SVR":
        r2_score(
            y_val,
            best_svr.predict(
                X_val
            )
        )
}

print("\nValidation R² scores:")

for name, score in validation_scores.items():

    print(
        f"{name:20s}: "
        f"{score:.4f} "
        f"({score * 100:.2f}%)"
    )

best_model_name = max(
    validation_scores,
    key=validation_scores.get
)

best_model = models[
    best_model_name
]

print(
    "\nBest model based on validation data:"
)

print(
    best_model_name
)

# Evaluate each tuned model on the held-out test set
print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

dt_test_results = evaluate_model(
    best_decision_tree,
    X_test,
    y_test,
    "Decision Tree"
)

rf_test_results = evaluate_model(
    best_random_forest,
    X_test,
    y_test,
    "Random Forest"
)

svr_test_results = evaluate_model(
    best_svr,
    X_test,
    y_test,
    "SVR"
)

# Final test-set comparison
final_results = pd.DataFrame(
    [
        dt_test_results,
        rf_test_results,
        svr_test_results
    ]
)

final_results = (
    final_results
    .sort_values(
        "R2",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(final_results)

# Check whether each model meets the 82% R² target