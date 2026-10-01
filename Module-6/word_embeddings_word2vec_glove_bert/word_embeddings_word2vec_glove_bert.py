import re
import numpy as np
import pandas as pd
import torch
import gensim.downloader as api

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel


# Small labelled dataset used to compare the three embedding approaches
data = {
    "text": [
        "I really loved the movie!",
        "The film was terrible and boring.",
        "Absolutely fantastic acting.",
        "Worst movie I've ever watched.",
        "It was okay, nothing special.",
        "I enjoyed every moment!",
        "I hate this film"
    ],
    "label": [
        "positive",
        "negative",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative"
    ]
}

df = pd.DataFrame(data)

print("\nDATASET")
print(df)


# Keep two reviews aside for testing so the classifiers are evaluated on unseen examples
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=2,
    random_state=0
)


# Use the same simple tokenization for Word2Vec and GloVe so their inputs are comparable
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()


# Train Word2Vec only on the training reviews to avoid using test text when learning embeddings
tokenized_train = [
    tokenize(review)
    for review in X_train
]

word2vec_model = Word2Vec(
    sentences=tokenized_train,
    vector_size=100,
    window=5,
    min_count=1,
    workers=1,
    epochs=100,
    seed=42
)


def get_word2vec_embedding(text):
    words = tokenize(text)

    vectors = [
        word2vec_model.wv[word]
        for word in words
        if word in word2vec_model.wv
    ]

    if not vectors:
        return np.zeros(
            word2vec_model.vector_size
        )

    return np.mean(vectors, axis=0)


X_train_w2v = np.array([
    get_word2vec_embedding(text)
    for text in X_train
])

X_test_w2v = np.array([
    get_word2vec_embedding(text)
    for text in X_test
])


w2v_classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)

w2v_classifier.fit(
    X_train_w2v,
    y_train
)

w2v_predictions = w2v_classifier.predict(
    X_test_w2v
)

w2v_accuracy = accuracy_score(
    y_test,
    w2v_predictions
)


# GloVe supplies pretrained word vectors learned from a much larger external text corpus
print("\nLoading GloVe...")

glove_model = api.load(
    "glove-wiki-gigaword-100"
)


def get_glove_embedding(text):
    words = tokenize(text)

    vectors = [
        glove_model[word]
        for word in words
        if word in glove_model
    ]

    if not vectors:
        return np.zeros(
            glove_model.vector_size
        )

    return np.mean(vectors, axis=0)


X_train_glove = np.array([
    get_glove_embedding(text)
    for text in X_train
])

X_test_glove = np.array([
    get_glove_embedding(text)
    for text in X_test
])


glove_classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)

glove_classifier.fit(
    X_train_glove,
    y_train
)

glove_predictions = glove_classifier.predict(
    X_test_glove
)

glove_accuracy = accuracy_score(
    y_test,
    glove_predictions
)


# BERT creates contextual embeddings, so the same word can have different representations by context
print("\nLoading BERT...")

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

bert_model = AutoModel.from_pretrained(
    model_name
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

bert_model.to(device)
bert_model.eval()


def get_bert_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = bert_model(**inputs)

    cls_vector = (
        output
        .last_hidden_state[:, 0, :]
        .squeeze()
        .cpu()
        .numpy()
    )

    return cls_vector


X_train_bert = np.array([
    get_bert_embedding(text)
    for text in X_train
])

X_test_bert = np.array([
    get_bert_embedding(text)
    for text in X_test
])


bert_classifier = LogisticRegression(
    max_iter=2000,
    random_state=42
)

bert_classifier.fit(
    X_train_bert,
    y_train
)

bert_predictions = bert_classifier.predict(
    X_test_bert
)

bert_accuracy = accuracy_score(
    y_test,
    bert_predictions
)


# Compare classification performance for each embedding method
print("\n==============================")
print("WORD2VEC")
print("==============================")

print("Accuracy:", w2v_accuracy)

print(
    classification_report(
        y_test,
        w2v_predictions,
        zero_division=0
    )
)


print("\n==============================")
print("GLOVE")
print("==============================")

print("Accuracy:", glove_accuracy)

print(
    classification_report(
        y_test,
        glove_predictions,
        zero_division=0
    )
)


print("\n==============================")
print("BERT")
print("==============================")

print("Accuracy:", bert_accuracy)

print(
    classification_report(
        y_test,
        bert_predictions,
        zero_division=0
    )
)


# Apply all three trained classifiers to the same new reviews for a direct comparison
new_reviews = [
    "The movie was incredible, I loved it!",
    "It was boring and too long.",
    "Mediocre experience, not bad."
]


new_w2v = np.array([
    get_word2vec_embedding(review)
    for review in new_reviews
])

new_glove = np.array([
    get_glove_embedding(review)
    for review in new_reviews
])

new_bert = np.array([
    get_bert_embedding(review)
    for review in new_reviews
])


w2v_new_predictions = (
    w2v_classifier.predict(new_w2v)
)

glove_new_predictions = (
    glove_classifier.predict(new_glove)
)

bert_new_predictions = (
    bert_classifier.predict(new_bert)
)


prediction_table = pd.DataFrame({
    "Review": new_reviews,
    "Word2Vec": w2v_new_predictions,
    "GloVe": glove_new_predictions,
    "BERT": bert_new_predictions
})


print("\n==============================")
print("NEW REVIEW PREDICTIONS")
print("==============================")

print(
    prediction_table.to_string(
        index=False
    )
)


# Collect the test accuracies in one table to make the embedding methods easier to compare
comparison = pd.DataFrame({
    "Embedding": [
        "Word2Vec",
        "GloVe",
        "BERT"
    ],
    "Accuracy": [
        w2v_accuracy,
        glove_accuracy,
        bert_accuracy
    ]
})


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(comparison)