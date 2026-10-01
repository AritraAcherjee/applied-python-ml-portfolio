import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the local dataset
df = pd.read_csv("books.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# Basic data cleaning
# Remove extra spaces so column names are consistent when referenced later
df.columns = df.columns.str.strip()

# Invalid text values are changed to NaN so they can be removed during cleaning
df["average_rating"] = pd.to_numeric(
    df["average_rating"],
    errors="coerce"
)

df["ratings_count"] = pd.to_numeric(
    df["ratings_count"],
    errors="coerce"
)

# Keep rows usable even when the author name is missing
df["authors"] = df["authors"].fillna("Unknown")

# These fields are required by the recommenders, so incomplete rows are removed
df = df.dropna(
    subset=["title", "average_rating", "ratings_count"]
)

df = df.reset_index(drop=True)

# Popularity-based recommender
# The weighted score balances a book's rating with how many people rated it
#
# WR = (v / (v + m)) * R + (m / (v + m)) * C
# R = average rating of the book
# v = number of ratings for the book
# m = minimum number of ratings required
# C = average rating across all books

C = df["average_rating"].mean()

# Using the 90th percentile avoids ranking books highly from only a few ratings
m = df["ratings_count"].quantile(0.90)

print("\nAverage rating of all books (C):", C)
print("Minimum ratings required (m):", m)

# Only books with enough ratings are included in the popularity ranking
popular_books = df[
    df["ratings_count"] >= m
].copy()


def weighted_rating(row):
    """
    Calculate IMDb-style weighted rating.
    """

    R = row["average_rating"]
    v = row["ratings_count"]

    score = (
        (v / (v + m)) * R
        +
        (m / (v + m)) * C
    )

    return score


# Calculate one weighted popularity score for each eligible book
popular_books["weighted_score"] = popular_books.apply(
    weighted_rating,
    axis=1
)


def popularity_recommender(n=10):
    """
    Return the top n most popular books.
    """

    recommendations = (
        popular_books
        .sort_values(
            "weighted_score",
            ascending=False
        )
        .head(n)
    )

    return recommendations[
        [
            "title",
            "authors",
            "average_rating",
            "ratings_count",
            "weighted_score"
        ]
    ]


print("\nTOP 10 POPULAR BOOKS:")
print(popularity_recommender(10))

# Content-based recommender
# Authors are used as the content feature required by this recommender
tfidf = TfidfVectorizer(
    stop_words="english"
)

# TF-IDF turns author text into numerical vectors that can be compared
tfidf_matrix = tfidf.fit_transform(
    df["authors"]
)

print("\nTF-IDF matrix shape:")
print(tfidf_matrix.shape)

# Cosine similarity compares vector direction rather than raw vector size
# A higher score means the books have more similar author information
cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

print("\nCosine similarity matrix shape:")
print(cosine_sim.shape)

# Map each title to its row index for quick recommendation lookups
# Duplicate titles are dropped so each title points to one index
indices = pd.Series(
    df.index,
    index=df["title"]
).drop_duplicates()


def content_recommender(title, n=10):
    """
    Recommend n books with authors similar to the selected book.
    """

    if title not in indices:
        return f"Book '{title}' was not found in the dataset."

    idx = indices[title]

    # Compare the selected book with every other book in the dataset
    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    # Highest similarity scores should appear first
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # The first result is the selected book itself, so leave it out
    similarity_scores = similarity_scores[1:n + 1]

    book_indices = [
        item[0]
        for item in similarity_scores
    ]

    scores = [
        item[1]
        for item in similarity_scores
    ]

    recommendations = df.iloc[book_indices][
        [
            "title",
            "authors",
            "average_rating"
        ]
    ].copy()

    recommendations["similarity_score"] = scores

    return recommendations


# Example recommendation
print("\nCONTENT-BASED RECOMMENDATIONS:")
print(
    content_recommender(
        "Harry Potter and the Half-Blood Prince (Harry Potter  #6)",
        10
    )
)