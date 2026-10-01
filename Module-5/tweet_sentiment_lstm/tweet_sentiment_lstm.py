import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# Load the tweet dataset
df = pd.read_csv("text_emotion.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nOriginal sentiment counts:")
print(df["sentiment"].value_counts())


# Keep the five sentiment classes required for the project
# Restricting the target set also keeps the classification task consistent with the required output layer.
five_sentiments = [
    "neutral",
    "worry",
    "happiness",
    "sadness",
    "love"
]

df = df[df["sentiment"].isin(five_sentiments)].copy()

print("\nDataset after keeping 5 classes:")
print(df.shape)

print("\nFive sentiment classes:")
print(df["sentiment"].value_counts())


# Use tweet content as the model input and sentiment as the target
# tweet_id and author are not needed for the required prediction task.
X = df["content"].astype(str)
y = df["sentiment"]


# Neural networks require numerical targets, so encode the sentiment labels
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClass mapping:")

for number, label in enumerate(label_encoder.classes_):
    print(number, "=", label)


# One-hot encoding matches the five-unit softmax output used by the network
y_categorical = to_categorical(
    y_encoded,
    num_classes=5
)


# Split 70% for training and 30% for testing
# Stratification helps preserve the sentiment distribution across both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Convert tweet text into token sequences for the neural network
tokenizer = Tokenizer(
    oov_token="<OOV>"
)

# Learn vocabulary only from training data so the test set remains unseen
tokenizer.fit_on_texts(X_train)

# Add one extra index because token numbering starts above zero
vocabulary_size = len(tokenizer.word_index) + 1

print("\nVocabulary size:", vocabulary_size)


# Replace words with their learned integer token IDs
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)


# Use the longest training tweet to define a consistent sequence length
maximum_sequence_length = max(
    len(sequence)
    for sequence in X_train_sequences
)

print("Maximum sequence length:", maximum_sequence_length)


# Pad shorter tweets so every input sequence has the same length
X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=maximum_sequence_length,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=maximum_sequence_length,
    padding="post",
    truncating="post"
)

print("\nTraining input shape:", X_train_padded.shape)
print("Testing input shape:", X_test_padded.shape)


# Build the required recurrent neural network architecture
# The embedding layer learns compact numerical representations of words before the LSTM layers process sequence order.
model = Sequential()

model.add(
    Embedding(
        input_dim=vocabulary_size,
        output_dim=10
    )
)

# Keep the full sequence output because a second LSTM layer follows
model.add(
    LSTM(
        128,
        return_sequences=True
    )
)

model.add(
    LSTM(64)
)

model.add(
    Dense(
        100,
        activation="relu"
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        5,
        activation="softmax"
    )
)


# Categorical cross-entropy matches the one-hot targets and softmax output
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Inspect the final network structure before training
print("\nMODEL SUMMARY")
print("=" * 60)

model.summary()


# Train for the required 10 epochs using mini-batches of 256 samples
# A portion of the training data is held back for validation during training.
history = model.fit(
    X_train_padded,
    y_train,
    batch_size=256,
    epochs=10,
    validation_split=0.10
)


# Evaluate final performance on the held-out test set
test_loss, test_accuracy = model.evaluate(
    X_test_padded,
    y_test,
    verbose=1
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy Percentage: {:.2f}%".format(
    test_accuracy * 100
))


# Show a small sample of predicted and actual sentiments for easier inspection
predictions = model.predict(
    X_test_padded[:10]
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

actual_classes = np.argmax(
    y_test[:10],
    axis=1
)

print("\nSample Predictions")
print("=" * 60)

X_test_reset = X_test.reset_index(drop=True)

for i in range(10):

    predicted_sentiment = label_encoder.inverse_transform(
        [predicted_classes[i]]
    )[0]

    actual_sentiment = label_encoder.inverse_transform(
        [actual_classes[i]]
    )[0]

    print("\nTweet:", X_test_reset.iloc[i])
    print("Actual:", actual_sentiment)
    print("Predicted:", predicted_sentiment)