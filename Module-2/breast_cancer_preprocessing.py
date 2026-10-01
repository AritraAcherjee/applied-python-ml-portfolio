import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# Load the breast cancer dataset directly from the project CSV
df = pd.read_csv("data.csv")

print(df.head())
df.info()


# Drop columns with no usable values because they cannot contribute to the analysis
df = df.dropna(axis=1, how="all")

# Remove incomplete rows so scaling and later analysis use complete observations
df = df.dropna()


# The ID identifies a record but does not describe the tumour, so exclude it from the features
if "id" in df.columns:
    df = df.drop("id", axis=1)


# Convert the diagnosis labels to numbers so they can be used in later analysis
encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

# LabelEncoder maps Benign to 0 and Malignant to 1


# Keep diagnosis as the target while the remaining columns become numerical features
X = df.drop("diagnosis", axis=1)


# Standardize the features so measurements with larger units do not dominate the analysis

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Restore column names after scaling so the refined data stays easy to interpret
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


# Recombine the standardized features with the diagnosis label

data_refined = X_scaled

data_refined["diagnosis"] = df["diagnosis"].reset_index(drop=True)


# Save the cleaned and standardized data for the later breast cancer modeling assessment
data_refined.to_csv(
    "data_refined.csv",
    index=False
)

print("Saved as data_refined.csv")


# Compare several feature distributions and relationships across diagnosis classes

sns.pairplot(
    data_refined,
    vars=data_refined.columns[:5],
    hue="diagnosis"
)

plt.show()


# The heatmap makes strong positive and negative relationships between variables easier to spot

plt.figure(figsize=(14, 10))

sns.heatmap(
    data_refined.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()


# Box plots provide a quick view of spread and possible extreme values across several features

plt.figure(figsize=(14, 6))

sns.boxplot(
    data=data_refined.iloc[:, :10]
)

plt.xticks(rotation=90)

plt.title("Box Plot of Features")

plt.show()


# Violin plots show both the distribution shape and how it differs between diagnosis classes

features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean"
]

for feature in features:

    plt.figure(figsize=(8, 6))

    sns.violinplot(
        x="diagnosis",
        y=feature,
        data=data_refined
    )

    plt.title("Violin Plot of " + feature)

    plt.show()


print("\nViolin Plot Description")

print(
    "A violin plot shows the distribution and density of numerical data. "
    "The wider parts of the violin represent areas where more data points "
    "are concentrated, while the narrow parts represent areas where fewer "
    "points occur."
)


# Use the IQR rule to flag values that fall well outside the middle 50% of each feature

print("\nOUTLIER ANALYSIS")

for feature in features:

    Q1 = data_refined[feature].quantile(0.25)

    Q3 = data_refined[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR

    outliers = data_refined[
        (data_refined[feature] < lower_limit)
        |
        (data_refined[feature] > upper_limit)
    ]

    print(
        feature,
        "has",
        len(outliers),
        "possible outliers."
    )


print(
    "\nThe violin plots show the shape and spread of each feature. "
    "Values extending further away from the main concentration of the "
    "distribution are potential outliers. The IQR calculation "
    "above is used to confirm how many possible outliers are present."
)