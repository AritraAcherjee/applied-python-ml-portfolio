import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("KaggleV2-May-2016.csv")

print("\nOriginal dataset shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())

# Remove accidental spaces from column names so later references are consistent
df.columns = df.columns.str.strip()

# Remove repeated and incomplete records before selecting model features
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Use the patient and appointment attributes required for this classification task
features = [
    "Gender",
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received"
]

target = "No-show"


# Check required columns early so a missing field produces a clear error instead of failing later
required_columns = features + [target]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nThe following required columns were not found: {missing_columns}\n"
        f"Available columns are: {df.columns.tolist()}"
    )


df = df[required_columns].copy()

print("\nColumns selected for the project:")
print(df.columns.tolist())


# Negative ages are not valid patient ages, so exclude those records
df = df[df["Age"] >= 0].copy()


# Standardize text labels before encoding so values differing only by case or spaces match
df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()


df["No-show"] = df["No-show"].astype(str).str.strip().str.upper()


print("\nOriginal target values:")
print(df["No-show"].value_counts())

# Convert gender labels to numeric values because the classifiers require numerical input
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

print("\nGender encoding:")
for original_value, encoded_value in zip(
    gender_encoder.classes_,
    gender_encoder.transform(gender_encoder.classes_)
):
    print(f"{original_value} = {encoded_value}")


# Encode the target so 1 represents a missed appointment and 0 represents attendance
target_mapping = {
    "NO": 0,
    "YES": 1
}

df["No-show"] = df["No-show"].map(target_mapping)


df.dropna(subset=["No-show"], inplace=True)


df["No-show"] = df["No-show"].astype(int)


# Coerce feature columns to numeric values because both tree models expect numerical input
for column in features:
    df[column] = pd.to_numeric(df[column], errors="coerce")


df.dropna(inplace=True)

print("\nCleaned dataset shape:", df.shape)

print("\nEncoded target distribution:")
print(df["No-show"].value_counts())

print("\nTarget meaning:")
print("0 = Patient attended the appointment")
print("1 = Patient did not attend the appointment")


# Separate predictors from the no-show outcome the models will learn to classify
X = df[features]
y = df[target]

print("\nFeature data shape:", X.shape)
print("Target data shape:", y.shape)


# Reserve 20% of the data while stratification keeps the no-show ratio similar across splits
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Divide the reserved data equally into validation and test sets, giving an 80/10/10 split
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)
print(f"Training observations:   {len(X_train)}")
print(f"Validation observations: {len(X_validation)}")
print(f"Testing observations:    {len(X_test)}")

print(
    "\nTraining percentage:",
    round(len(X_train) / len(X) * 100, 2),
    "%"
)

print(
    "Validation percentage:",
    round(len(X_validation) / len(X) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(X_test) / len(X) * 100, 2),
    "%"
)



print("\n" + "=" * 70)
print("DECISION TREE RESULTS")
print("=" * 70)


# Compare several split criteria on validation data rather than choosing from test performance
decision_tree_criteria = ["gini", "entropy", "log_loss"]

best_tree_model = None
best_tree_criterion = None
best_tree_validation_accuracy = 0

for criterion in decision_tree_criteria:
    try:
        tree_model = DecisionTreeClassifier(
            criterion=criterion,
            random_state=42
        )

        tree_model.fit(X_train, y_train)

        training_predictions = tree_model.predict(X_train)
        validation_predictions = tree_model.predict(X_validation)

        training_accuracy = accuracy_score(
            y_train,
            training_predictions
        )

        validation_accuracy = accuracy_score(
            y_validation,
            validation_predictions
        )

        print(f"\nCriterion: {criterion}")
        print(f"Training accuracy:   {training_accuracy:.4f}")
        print(f"Validation accuracy: {validation_accuracy:.4f}")

        # Use validation accuracy for model selection so the test set remains unseen
        if validation_accuracy > best_tree_validation_accuracy:
            best_tree_validation_accuracy = validation_accuracy
            best_tree_model = tree_model
            best_tree_criterion = criterion

    except ValueError:
        print(
            f"\nCriterion '{criterion}' is not supported "
            "by your installed scikit-learn version."
        )



# Evaluate the selected Decision Tree on the held-out test set only after model selection
tree_test_predictions = best_tree_model.predict(X_test)
tree_test_accuracy = accuracy_score(y_test, tree_test_predictions)

print("\n" + "-" * 70)
print("BEST DECISION TREE")
print("-" * 70)
print(f"Best criterion: {best_tree_criterion}")
print(
    f"Best validation accuracy: "
    f"{best_tree_validation_accuracy:.4f}"
)
print(f"Final test accuracy: {tree_test_accuracy:.4f}")

print("\nDecision Tree confusion matrix:")
tree_confusion_matrix = confusion_matrix(
    y_test,
    tree_test_predictions
)
print(tree_confusion_matrix)

print("\nDecision Tree classification report:")
print(
    classification_report(
        y_test,
        tree_test_predictions,
        target_names=["Attended", "No-show"],
        zero_division=0
    )
)



print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)


# Compare several forest sizes to see which number of trees performs best on validation data
estimator_values = [10, 25, 50, 100, 200]

best_forest_model = None
best_estimator_value = None
best_forest_validation_accuracy = 0

for estimator_count in estimator_values:
    forest_model = RandomForestClassifier(
        n_estimators=estimator_count,
        random_state=42,
        n_jobs=-1
    )

    forest_model.fit(X_train, y_train)

    training_predictions = forest_model.predict(X_train)
    validation_predictions = forest_model.predict(X_validation)

    training_accuracy = accuracy_score(
        y_train,
        training_predictions
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    print(f"\nNumber of estimators: {estimator_count}")
    print(f"Training accuracy:   {training_accuracy:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")


    # Keep the forest size with the strongest validation accuracy
    if validation_accuracy > best_forest_validation_accuracy:
        best_forest_validation_accuracy = validation_accuracy
        best_forest_model = forest_model
        best_estimator_value = estimator_count



# Test the selected Random Forest only after its estimator count has been chosen
forest_test_predictions = best_forest_model.predict(X_test)
forest_test_accuracy = accuracy_score(
    y_test,
    forest_test_predictions
)

print("\n" + "-" * 70)
print("BEST RANDOM FOREST")
print("-" * 70)
print(f"Best number of estimators: {best_estimator_value}")
print(
    f"Best validation accuracy: "
    f"{best_forest_validation_accuracy:.4f}"
)
print(f"Final test accuracy: {forest_test_accuracy:.4f}")

print("\nRandom Forest confusion matrix:")
forest_confusion_matrix = confusion_matrix(
    y_test,
    forest_test_predictions
)
print(forest_confusion_matrix)

print("\nRandom Forest classification report:")
print(
    classification_report(
        y_test,
        forest_test_predictions,
        target_names=["Attended", "No-show"],
        zero_division=0
    )
)



# Feature importance shows which inputs contributed most to the selected Random Forest
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": best_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nRandom Forest feature importance:")
print(feature_importance.to_string(index=False))



print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    f"Decision Tree test accuracy: "
    f"{tree_test_accuracy:.4f}"
)

print(
    f"Random Forest test accuracy: "
    f"{forest_test_accuracy:.4f}"
)

# Compare final test accuracy after both model choices have already been fixed
if forest_test_accuracy > tree_test_accuracy:
    print("\nThe Random Forest produced the highest test accuracy.")
    overall_best_model = "Random Forest"

elif tree_test_accuracy > forest_test_accuracy:
    print("\nThe Decision Tree produced the highest test accuracy.")
    overall_best_model = "Decision Tree"

else:
    print("\nBoth models produced the same test accuracy.")
    overall_best_model = "Tie"

print(f"Overall best model: {overall_best_model}")

print("\nProgram completed successfully.")