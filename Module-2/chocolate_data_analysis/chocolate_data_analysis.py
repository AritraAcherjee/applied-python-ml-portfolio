import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("flavors_of_cacao.csv")

# Remove inconsistent spacing from column names so they are easier to reference reliably
df.columns = [" ".join(column.split()) for column in df.columns]


# Treat non-breaking-space placeholders as missing values before checking data quality
df = df.replace("\xa0", pd.NA)

print("Column names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of tuples:")
print(len(df))

print("\nNumber of unique company names:")
print(df["Company (Maker-if known)"].nunique())

# Filter by review year to answer the 2013 review-count question
reviews_2013 = df[df["Review Date"] == 2013]

print("\nNumber of reviews made in 2013:")
print(len(reviews_2013))


print("\nMissing values in Bean Type:")
print(df["Bean Type"].isnull().sum())

# Remove incomplete rows so later analysis and transformations use complete observations
df = df.dropna()

print("\nNumber of rows after removing missing values:")
print(len(df))

plt.hist(df["Rating"], bins=10, edgecolor="black")

plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.title("Distribution of Chocolate Ratings")

plt.show()


# Remove the percent sign so cocoa content can be treated as a numerical variable
df["Cocoa Percent"] = (
    df["Cocoa Percent"]
    .str.replace("%", "", regex=False)
    .astype(float)
)

print("\nConverted Cocoa Percent:")
print(df["Cocoa Percent"].head())

print("\nCocoa Percent data type:")
print(df["Cocoa Percent"].dtype)

plt.hist(df["Cocoa Percent"], bins=10, edgecolor="black")

plt.xlabel("Cocoa Percent")
plt.ylabel("Frequency")
plt.title("Distribution of Cocoa Percent")

plt.show()

plt.scatter(
    df["Cocoa Percent"],
    df["Rating"],
    alpha=0.1
)

plt.xlabel("Cocoa Percent")
plt.ylabel("Rating")
plt.title("Cocoa Percent vs Rating")

plt.show()

# Correlation summarizes the direction and strength of the linear relationship between cocoa content and rating
correlation = df["Cocoa Percent"].corr(df["Rating"])

print("\nCorrelation between Cocoa Percent and Rating:")
print(correlation)

if correlation > 0.3:
    print("There appears to be a positive relationship.")
elif correlation < -0.3:
    print("There appears to be a negative relationship.")
else:
    print("There appears to be little or no strong relationship.")


# Min-max scaling converts ratings to a 0-to-1 range while preserving their relative order
scaler = MinMaxScaler()

df["Normalized Rating"] = scaler.fit_transform(
    df[["Rating"]]
)

print("\nOriginal and Normalized Ratings:")
print(
    df[
        ["Rating", "Normalized Rating"]
    ].head(20)
)

# Average ratings provide a simple way to compare companies across all of their reviewed products
company_average = (
    df.groupby("Company (Maker-if known)")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

print("\nCompanies ordered by average rating:")
print(company_average)


# Categorical company and location values are converted to numeric indicator columns
encoder = OneHotEncoder(
    # Ignore categories that may appear later but were not present when the encoder was fitted
    handle_unknown="ignore",
    sparse_output=False
)

encoded_data = encoder.fit_transform(
    df[
        [
            "Company (Maker-if known)",
            "Company Location"
        ]
    ]
)

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(
        [
            "Company (Maker-if known)",
            "Company Location"
        ]
    ),
    index=df.index
)


print("\nEncoded Company and Location Data:")
print(encoded_df.head())


print("\nFinal Dataset:")
print(df.head())


print("\nFinal Dataset Information:")
df.info()