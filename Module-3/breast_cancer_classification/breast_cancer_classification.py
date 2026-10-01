import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif


df = pd.read_csv("data_refined.csv")
print("Dataset loaded successfully")
print("\nFirst five rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)


# Diagnosis is the classification target the models will learn to predict
y = df["diagnosis"]


# Absolute correlation is used so strong positive and negative relationships are both considered
correlations = (df.corr()["diagnosis"].abs().sort_values(ascending=False))

print("\nCorrelations with diagnosis:")
print(correlations)


# Keep features with a reasonably strong relationship to diagnosis for the reduced feature set
correlation_limit = 0.50


selected_features = correlations[
    (correlations >= correlation_limit)
    &
    (correlations.index != "diagnosis")
].index.tolist()


print("\nCorrelation selected features:")
print(selected_features)


# Full-feature input excludes only the target column
X_full = df.drop(
    columns=["diagnosis"]
)


X_correlation = df[
    selected_features
]



selector = SelectKBest(
    score_func=f_classif,
    k=10
)


selector.fit(
    X_full,
    y
)


second_selected_features = X_full.columns[
    selector.get_support()
].tolist()


print("\nSelectKBest selected features:")
print(second_selected_features)


X_selectkbest = df[
    second_selected_features
]


# Use the same evaluation process for each feature set so model comparisons stay consistent
def test_models(X, y, feature_set_name):

    print("\n")
    print("=" * 60)
    print(feature_set_name)
    print("=" * 60)


    # Reserve 20% of the data while stratification keeps class proportions similar across splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # Split the reserved data equally so validation guides comparison while test data stays unseen
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )


    print("\nTraining rows:")
    print(len(X_train))

    print("Validation rows:")
    print(len(X_validation))

    print("Testing rows:")
    print(len(X_test))


    # Search several neighborhood sizes because KNN performance depends strongly on k
    knn_parameters = {
        "n_neighbors": range(1, 21)
    }


    # Five-fold cross-validation selects k using only the training data
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_parameters,
        cv=5,
        scoring="accuracy"
    )


    knn_grid.fit(
        X_train,
        y_train
    )


    print("\nBest K for KNN:")
    print(
        knn_grid.best_params_
    )


    print(
        "Best KNN cross-validation accuracy:"
    )

    print(
        knn_grid.best_score_
    )


    best_knn_model = (
        knn_grid.best_estimator_
    )


    # Random Forest provides a tree-based comparison that can capture nonlinear relationships
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )


    # The RBF kernel lets SVC model nonlinear decision boundaries between diagnosis classes
    svc_model = SVC(
        kernel="rbf",
        C=1,
        gamma="scale"
    )


    models = {

        "K-Nearest Neighbors":
            best_knn_model,

        "Random Forest":
            random_forest_model,

        "Support Vector Classifier":
            svc_model
    }


    print("\nVALIDATION RESULTS")


    # Compare all models on the same validation split before reviewing final test performance
    for model_name, model in models.items():

        model.fit(
            X_train,
            y_train
        )


        validation_predictions = model.predict(
            X_validation
        )


        validation_accuracy = accuracy_score(
            y_validation,
            validation_predictions
        )


        print("\n-----------------------------------")
        print(model_name)
        print("-----------------------------------")


        print(
            "Validation accuracy:",
            validation_accuracy
        )


        print(
            "\nValidation confusion matrix:"
        )


        print(
            confusion_matrix(
                y_validation,
                validation_predictions
            )
        )


    # Final test results show how the fitted models perform on data not used for validation
    print("\nFINAL TEST RESULTS")


    for model_name, model in models.items():

        model.fit(
            X_train,
            y_train
        )


        test_predictions = model.predict(
            X_test
        )


        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )


        print("\n-----------------------------------")
        print(model_name)
        print("-----------------------------------")


        print(
            "Test accuracy:",
            test_accuracy
        )


        print(
            "\nTest confusion matrix:"
        )


        print(
            confusion_matrix(
                y_test,
                test_predictions
            )
        )


        # Check whether the model meets the project accuracy requirement
        if test_accuracy >= 0.94:

            print(
                "Accuracy requirement met."
            )

        else:

            print(
                "Accuracy is below 94%."
            )


# Evaluate the complete feature set first
test_models(
    X_full,
    y,
    "FULL FEATURE DATASET"
)


# Compare performance after correlation-based feature reduction
test_models(
    X_correlation,
    y,
    "CORRELATION REDUCED DATASET"
)


# Compare performance with the ten features selected by ANOVA F-score
test_models(
    X_selectkbest,
    y,
    "SELECT K BEST REDUCED DATASET"
)