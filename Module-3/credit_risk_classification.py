import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report



credit_data = fetch_openml(
    name="credit-g",
    version=1,
    as_frame=True
)


# Work with a copy so the downloaded dataset object is left unchanged
df = credit_data.frame.copy()

print("First five rows of the original dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset column names:")
print(df.columns.tolist())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nTotal number of missing values:")
print(df.isnull().sum().sum())


# Remove incomplete rows so preprocessing and KNN receive usable feature values
df = df.dropna()

print("\nDataset shape after removing missing values:")
print(df.shape)


# Separate numerical and categorical predictors because they require different preprocessing
numeric_features = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "age"
]


nominal_features = [
    "checking_status",
    "purpose",
    "housing"
]


selected_features = numeric_features + nominal_features


# Use the selected credit attributes as predictors and the class column as the target
X = df[selected_features]

y = df["class"]

print("\nSelected numeric features:")
print(numeric_features)

print("\nSelected nominal features:")
print(nominal_features)

print("\nPreview of selected features:")
print(X.head())

print("\nTarget class counts:")
print(y.value_counts())


# Scaling is important for KNN because distance calculations are sensitive to feature magnitude
numeric_transformer = StandardScaler()


# Convert categories to numeric indicators and safely handle unseen values later
nominal_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# Apply scaling and one-hot encoding to the appropriate columns in one preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "nominal",
            nominal_transformer,
            nominal_features
        )
    ]
)


# Reserve 20% of the data while stratification keeps the class balance similar across splits
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Divide the reserved data equally so validation selects k while the test set stays unseen
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nData split sizes:")
print("Training observations:", len(X_train))
print("Validation observations:", len(X_validation))
print("Testing observations:", len(X_test))

print("\nData split percentages:")
print(f"Training:   {len(X_train) / len(X) * 100:.1f}%")
print(f"Validation: {len(X_validation) / len(X) * 100:.1f}%")
print(f"Testing:    {len(X_test) / len(X) * 100:.1f}%")


# Store validation accuracy for each candidate k so the neighborhood size can be compared
validation_results = []


# Try several neighborhood sizes because KNN performance can change substantially with k
for k in range(1, 26):
    knn_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                KNeighborsClassifier(n_neighbors=k)
            )
        ]
    )

   
    knn_pipeline.fit(X_train, y_train)

    
    validation_predictions = knn_pipeline.predict(X_validation)

    
    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

   
    validation_results.append(
        {
            "k": k,
            "validation_accuracy": validation_accuracy
        }
    )

    print(
        f"k = {k:2d} | "
        f"Validation accuracy = {validation_accuracy:.4f}"
    )


results_df = pd.DataFrame(validation_results)

print("\nValidation results:")
print(results_df)


# Choose k using validation accuracy rather than test performance
highest_validation_accuracy = results_df[
    "validation_accuracy"
].max()


# If several k values tie, use the smallest one for a consistent selection rule
best_k = int(results_df[results_df["validation_accuracy"] == highest_validation_accuracy]["k"].min())

print("\nBest KNN settings:")
print("Best k:", best_k)
print(
    "Best validation accuracy:",
    round(highest_validation_accuracy, 4)
)


plt.figure(figsize=(10, 5))

plt.plot(
    results_df["k"],
    results_df["validation_accuracy"],
    marker="o"
)

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Validation Accuracy")
plt.title("KNN Validation Accuracy for Different Values of k")
plt.xticks(range(1, 26))
plt.grid(True)
plt.tight_layout()
plt.show()


# Build the final KNN pipeline with the k value selected from validation data
final_knn_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=best_k)
        )
    ]
)


final_knn_model.fit(X_train, y_train)


# Evaluate on the held-out test set only after the model settings have been chosen
test_predictions = final_knn_model.predict(X_test)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\nFINAL TEST RESULTS")
print("------------------")
print("Best k:", best_k)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test accuracy percentage: {test_accuracy * 100:.2f}%")


# The confusion matrix shows which credit classes are being confused with one another
confusion_matrix_result = confusion_matrix(
    y_test,
    test_predictions,
    labels=final_knn_model.classes_
)

print("\nConfusion matrix:")
print(confusion_matrix_result)

ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix_result,
    display_labels=final_knn_model.classes_
).plot()

plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.show()



print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)
