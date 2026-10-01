import re
import string
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Download the language resources used by the tokenizer, stopword list, and lemmatizer
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Sample text includes punctuation, emojis, and a URL so each preprocessing step can be demonstrated
raw_text = """
Wow!!! I just watched the new sci-fi movie 🚀.
It was AMAZING 😍!
Check it out at https://example.com/movie
"""


print("=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(raw_text)


# Lowercasing keeps words such as "Amazing" and "amazing" from being treated as different tokens
lowercase_text = raw_text.lower()

print("\n" + "=" * 60)
print("STEP 1 - LOWERCASE TEXT")
print("=" * 60)
print(lowercase_text)


# Clean text before tokenization
# URLs do not contribute useful language information for this example
cleaned_text = re.sub(r"http\S+|www\S+|https\S+", "", lowercase_text)

# Keep ASCII text only so emojis and other non-ASCII symbols do not become unwanted tokens
cleaned_text = cleaned_text.encode("ascii", "ignore").decode()

# Punctuation is removed so the remaining tokens focus on the words themselves
cleaned_text = cleaned_text.translate(
    str.maketrans("", "", string.punctuation)
)

# Cleaning can leave repeated whitespace, so collapse it to a single space
cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

print("\n" + "=" * 60)
print("STEP 2 - CLEANED TEXT")
print("=" * 60)
print(cleaned_text)


# Split the cleaned text into sentence-level units
sentence_tokens = sent_tokenize(cleaned_text)

print("\n" + "=" * 60)
print("STEP 3 - SENTENCE TOKENS")
print("=" * 60)

for index, sentence in enumerate(sentence_tokens, start=1):
    print(f"Sentence {index}: {sentence}")


# Break the cleaned text into individual word tokens
word_tokens = word_tokenize(cleaned_text)

print("\n" + "=" * 60)
print("STEP 4 - WORD TOKENS")
print("=" * 60)
print(word_tokens)


# Remove common English words that usually carry little meaning for text analysis
stop_words = set(stopwords.words("english"))

filtered_tokens = [
    word for word in word_tokens
    if word not in stop_words
]

print("\n" + "=" * 60)
print("STEP 5 - TOKENS AFTER STOPWORD REMOVAL")
print("=" * 60)
print(filtered_tokens)


# Stemming reduces related words to simpler word stems
stemmer = PorterStemmer()

stemmed_words = [
    stemmer.stem(word)
    for word in filtered_tokens
]

print("\n" + "=" * 60)
print("STEP 6 - STEMMED WORDS")
print("=" * 60)
print(stemmed_words)


# Lemmatization reduces words to dictionary-style base forms while preserving more linguistic meaning
lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word)
    for word in filtered_tokens
]

print("\n" + "=" * 60)
print("STEP 7 - LEMMATIZED WORDS")
print("=" * 60)
print(lemmatized_words)


print("\n" + "=" * 60)
print("TEXT PREPROCESSING COMPLETED")
print("=" * 60)
