# Sentiment Analysis Using Bag-of-Words and TF-IDF

# Libraries used for splitting, vectorizing, training, and evaluation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


# Small labelled review set used to compare the two text representations
reviews = [
    "I love this phone, it works perfectly!",
    "Terrible service, I will never buy again.",
    "Not bad, but could be improved.",
    "Absolutely fantastic experience.",
    "Worst product ever!"
]

labels = [
    "positive",
    "negative",
    "neutral",
    "positive",
    "negative"
]


# Put the reviews and labels together so the original data is easy to inspect
dataset = pd.DataFrame({
    "Review": reviews,
    "Sentiment": labels
})

print("ORIGINAL DATASET")
print(dataset)
print()


# Keep 20% for testing and fix the random state so the split is reproducible
X_train, X_test, y_train, y_test = train_test_split(
    reviews,
    labels,
    test_size=0.20,
    random_state=42
)

print("=" * 60)
print("TRAINING DATA")
print("=" * 60)

for review, label in zip(X_train, y_train):
    print(f"{label:10} -> {review}")

print()

print("=" * 60)
print("TESTING DATA")
print("=" * 60)

for review, label in zip(X_test, y_test):
    print(f"{label:10} -> {review}")

print()


# Bag-of-Words representation
print("\n" + "=" * 60)
print("BAG-OF-WORDS REPRESENTATION")
print("=" * 60)


# Bag-of-Words represents each review by how often its words appear
bow_vectorizer = CountVectorizer()

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)


# These are the words learned from the training reviews
bow_features = bow_vectorizer.get_feature_names_out()

print("\nBag-of-Words Features:")
print(bow_features)


# Convert the sparse matrix to a DataFrame so the word counts are easier to inspect
bow_matrix = pd.DataFrame(
    X_train_bow.toarray(),
    columns=bow_features
)

print("\nBag-of-Words Feature Matrix:")
print(bow_matrix)


# Train a classifier on the Bag-of-Words features
bow_model = LogisticRegression(max_iter=1000)

bow_model.fit(X_train_bow, y_train)


# The test reviews use the vocabulary learned from the training data
bow_predictions = bow_model.predict(X_test_bow)


# Evaluate the Bag-of-Words predictions against the unseen test labels
bow_accuracy = accuracy_score(y_test, bow_predictions)

print("\nActual Test Labels:")
print(y_test)

print("\nBag-of-Words Predictions:")
print(bow_predictions)

print("\nBag-of-Words Accuracy:")
print(bow_accuracy)

print("\nBag-of-Words Classification Report:")
print(
    classification_report(
        y_test,
        bow_predictions,
        zero_division=0
    )
)


# TF-IDF representation
print("\n" + "=" * 60)
print("TF-IDF REPRESENTATION")
print("=" * 60)


# TF-IDF reduces the influence of very common words and emphasizes more distinctive terms
tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# Show the vocabulary learned by the TF-IDF vectorizer
tfidf_features = tfidf_vectorizer.get_feature_names_out()

print("\nTF-IDF Features:")
print(tfidf_features)


# The matrix shows the weighted importance of each word in each training review
tfidf_matrix = pd.DataFrame(
    X_train_tfidf.toarray(),
    columns=tfidf_features
)

print("\nTF-IDF Feature Matrix:")
print(tfidf_matrix.round(3))


# Train the same classifier again so only the text representation changes
tfidf_model = LogisticRegression(max_iter=1000)

tfidf_model.fit(X_train_tfidf, y_train)


# Transform the test reviews with the fitted TF-IDF vocabulary before prediction
tfidf_predictions = tfidf_model.predict(X_test_tfidf)


# Evaluate TF-IDF using the same test labels for a fair comparison
tfidf_accuracy = accuracy_score(
    y_test,
    tfidf_predictions
)

print("\nActual Test Labels:")
print(y_test)

print("\nTF-IDF Predictions:")
print(tfidf_predictions)

print("\nTF-IDF Accuracy:")
print(tfidf_accuracy)

print("\nTF-IDF Classification Report:")
print(
    classification_report(
        y_test,
        tfidf_predictions,
        zero_division=0
    )
)


# Predictions for new reviews
print("\n" + "=" * 60)
print("PREDICTIONS FOR NEW REVIEWS")
print("=" * 60)

new_reviews = [
    "I really enjoyed the product, very satisfied!",
    "The product is bad and disappointing."
]


# Reuse the fitted Bag-of-Words vectorizer so new reviews match the training feature space
new_reviews_bow = bow_vectorizer.transform(new_reviews)

new_bow_predictions = bow_model.predict(
    new_reviews_bow
)


# Apply the fitted TF-IDF transformation before using the TF-IDF classifier
new_reviews_tfidf = tfidf_vectorizer.transform(
    new_reviews
)

new_tfidf_predictions = tfidf_model.predict(
    new_reviews_tfidf
)


# Compare how both trained models classify the same new reviews
for i in range(len(new_reviews)):

    print(f"\nReview: {new_reviews[i]}")

    print(
        "Bag-of-Words Prediction:",
        new_bow_predictions[i]
    )

    print(
        "TF-IDF Prediction:",
        new_tfidf_predictions[i]
    )


# Compare the two text representations on this train-test split
print("\n" + "=" * 60)
print("BAG-OF-WORDS VS TF-IDF COMPARISON")
print("=" * 60)

print(
    f"Bag-of-Words Accuracy: {bow_accuracy:.2f}"
)

print(
    f"TF-IDF Accuracy:       {tfidf_accuracy:.2f}"
)


if bow_accuracy > tfidf_accuracy:

    print(
        "\nBag-of-Words performed better "
        "on this train-test split."
    )

elif tfidf_accuracy > bow_accuracy:

    print(
        "\nTF-IDF performed better "
        "on this train-test split."
    )

else:

    print(
        "\nBoth methods achieved the same accuracy "
        "on this train-test split."
    )


print("""
Explanation:

Bag-of-Words represents documents using the number of
times each word appears.

TF-IDF also considers word frequency, but gives lower
importance to words that occur very frequently and more
importance to words that are relatively distinctive.

TF-IDF often performs better on larger text datasets
because important words receive more meaningful weights.

However, this dataset contains only five reviews. With an
80/20 split, the test set contains only one review.
Therefore, the accuracy score is not reliable enough to
make a strong conclusion about which method is generally
better.
""")
