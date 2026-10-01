import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, dendrogram


RANDOM_STATE = 42  # Keeps K-Means results reproducible

K_MIN = 2
K_MAX = 10

FEATURES = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

# Load the wholesale customer dataset directly from the project CSV
df = pd.read_csv("Wholesale customers data.csv")

print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nStatistical summary:")
print(df.describe())

print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

# Check data quality before clustering so missing values do not distort distance calculations
missing_values = df.isnull().sum()

print("\nMissing values:")
print(missing_values)


if missing_values.sum() > 0:

    print(
        "\nMissing values found. "
        "Replacing numeric missing values with medians."
    )

    numeric_columns = (
        df.select_dtypes(include=np.number).columns
    )

    for column in numeric_columns:

        df[column] = (
            # Median is less affected by extreme spending values than the mean
            df[column]
            .fillna(df[column].median())
        )

else:

    print("\nNo missing values detected.")


# Duplicate customers could give repeated observations extra influence on the clusters
duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)


if duplicates > 0:

    df = df.drop_duplicates().reset_index(drop=True)

    print("Duplicates removed.")


# Cluster customers using product spending only; Channel and Region are excluded from the feature space
X = df[FEATURES].copy()


print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

print("\nFeatures selected:")

for feature in FEATURES:
    print("-", feature)

print("\nExcluded variables: Channel and Region")

# Scaling keeps high-spending categories from dominating Euclidean distance simply because of their units
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=FEATURES
)

print("\n" + "=" * 70)
print("STANDARDIZED FEATURES")
print("=" * 70)

print(scaled_df.head())


# Higher Dunn Index values indicate clusters that are compact internally and well separated from one another
def dunn_index(data, labels):


    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    max_intra = 0.0

    for label in unique_labels:

        cluster_points = data[
            labels == label
        ]

        if len(cluster_points) > 1:

            distances = pdist(
                cluster_points
            )

            if len(distances) > 0:

                diameter = np.max(
                    distances
                )

                max_intra = max(
                    max_intra,
                    diameter
                )


    min_inter = np.inf

    for i in range(
        len(unique_labels)
    ):

        for j in range(
            i + 1,
            len(unique_labels)
        ):

            cluster_a = data[
                labels == unique_labels[i]
            ]

            cluster_b = data[
                labels == unique_labels[j]
            ]

            distances = cdist(
                cluster_a,
                cluster_b
            )

            current_min = np.min(
                distances
            )

            min_inter = min(
                min_inter,
                current_min
            )

    if max_intra == 0:
        return np.nan

    return min_inter / max_intra

k_values = list(
    range(
        K_MIN,
        K_MAX + 1
    )
)

inertia_values = []
dunn_values = []

models = {}

print("\n" + "=" * 70)
print("K-MEANS MODEL TESTING")
print("=" * 70)

# Compare several cluster counts rather than choosing K arbitrarily
for k in k_values:

    # Multiple initializations reduce the chance of keeping a poor random centroid starting point
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    inertia = model.inertia_

    dunn = dunn_index(
        X_scaled,
        labels
    )

    inertia_values.append(
        inertia
    )

    dunn_values.append(
        dunn
    )

    models[k] = model

    print(
        f"K = {k:2d} | "
        f"Inertia = {inertia:10.2f} | "
        f"Dunn Index = {dunn:.6f}"
    )


metrics_df = pd.DataFrame({

    "K": k_values,

    "Inertia": inertia_values,

    "Dunn Index": dunn_values

})

print("\n" + "=" * 70)
print("CLUSTERING METRICS")
print("=" * 70)

print(
    metrics_df.to_string(
        index=False
    )
)


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    k_values,
    inertia_values,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia / WCSS"
)

plt.title(
    "Elbow Method for K-Means"
)

plt.xticks(
    k_values
)

plt.grid(True)

plt.tight_layout()

plt.show()


# Estimate the elbow by finding the point farthest from the line joining the curve endpoints
def estimate_elbow(
    k_values,
    inertia_values
):


    x = np.array(
        k_values,
        dtype=float
    )

    y = np.array(
        inertia_values,
        dtype=float
    )


    x_normalized = (
        (x - x.min()) /
        (x.max() - x.min())
    )

    y_normalized = (
        (y - y.min()) /
        (y.max() - y.min())
    )


    points = np.column_stack(
        (
            x_normalized,
            y_normalized
        )
    )


    first_point = points[0]

    last_point = points[-1]

    line = (
        last_point -
        first_point
    )

    line_length = np.linalg.norm(
        line
    )


    distances = []

    for point in points:

        vector = (
            point -
            first_point
        )

        distance = abs(
            np.cross(
                line,
                vector
            )
        ) / line_length

        distances.append(
            distance
        )


    elbow_index = np.argmax(
        distances
    )

    return int(
        x[elbow_index]
    )


elbow_k = estimate_elbow(
    k_values,
    inertia_values
)


print("\nEstimated elbow K:", elbow_k)


elbow_position = (
    k_values.index(elbow_k)
)

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    k_values,
    inertia_values,
    marker="o"
)

plt.scatter(
    elbow_k,
    inertia_values[
        elbow_position
    ],
    s=180,
    label=f"Estimated Elbow: K={elbow_k}"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia / WCSS"
)

plt.title(
    "Elbow Curve with Estimated Optimal K"
)

plt.xticks(
    k_values
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    k_values,
    dunn_values,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Dunn Index"
)

plt.title(
    "Dunn Index for Different Values of K"
)

plt.xticks(
    k_values
)

plt.grid(True)

plt.tight_layout()

plt.show()


# Dunn Index provides a second view of cluster quality to compare with the elbow estimate
best_dunn_position = np.nanargmax(
    dunn_values
)

best_dunn_k = k_values[
    best_dunn_position
]

highest_dunn = dunn_values[
    best_dunn_position
]


print("\n" + "=" * 70)
print("MODEL-SELECTION INFORMATION")
print("=" * 70)

print(
    "Estimated elbow K:",
    elbow_k
)

print(
    "K with highest Dunn Index:",
    best_dunn_k
)

print(
    "Highest Dunn Index:",
    round(
        highest_dunn,
        6
    )
)


# Use the elbow estimate as the final K while still reporting the Dunn-based alternative
best_k = elbow_k


print(
    "\nFinal selected K =",
    best_k
)


final_model = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=10
)

final_labels = final_model.fit_predict(
    X_scaled
)


clustered_df = df.copy()

clustered_df[
    "Cluster"
] = final_labels


print("\n" + "=" * 70)
print("CLUSTERED DATASET")
print("=" * 70)

print(
    clustered_df.head(10)
)


first_10_predictions = (
    final_model.predict(
        X_scaled[:10]
    )
)


prediction_table = (
    X.iloc[:10].copy()
)

prediction_table[
    "Predicted Cluster"
] = first_10_predictions


print("\n" + "=" * 70)
print("FIRST 10 CUSTOMER PREDICTIONS")
print("=" * 70)

print(
    prediction_table.to_string()
)


print("\nIndividual predictions:")

for customer_number, cluster in enumerate(
    first_10_predictions,
    start=1
):

    print(
        f"Customer {customer_number:2d}"
        f" -> Cluster {cluster}"
    )


cluster_sizes = (
    clustered_df[
        "Cluster"
    ]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 70)
print("CLUSTER SIZES")
print("=" * 70)

print(
    cluster_sizes
)


# Convert cluster centers back to the original spending scale so they are easier to interpret
centers_original = (
    scaler.inverse_transform(
        final_model.cluster_centers_
    )
)


centers_df = pd.DataFrame(
    centers_original,
    columns=FEATURES
)


centers_df.index = [

    f"Cluster {i}"

    for i in range(
        best_k
    )
]


print("\n" + "=" * 70)
print("CLUSTER CENTERS - ORIGINAL SCALE")
print("=" * 70)

print(
    centers_df.round(2)
)


# Average spending by cluster helps describe the customer segments in practical terms
cluster_profile = (

    clustered_df
    .groupby("Cluster")[FEATURES]
    .mean()

)

print("\n" + "=" * 70)
print("AVERAGE CUSTOMER SPENDING BY CLUSTER")
print("=" * 70)

print(
    cluster_profile.round(2)
)


cluster_profile.T.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.xlabel(
    "Product Category"
)

plt.ylabel(
    "Average Annual Spending"
)

plt.title(
    "Average Spending Profile by Cluster"
)

plt.xticks(
    rotation=45
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# PCA reduces the six standardized features to two dimensions for visualization only
pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

plt.figure(
    figsize=(10, 7)
)

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=final_labels,
    cmap="viridis",
    alpha=0.75
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    f"K-Means Customer Segmentation (K={best_k})"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(True)

plt.tight_layout()

plt.show()

final_dunn = dunn_index(
    X_scaled,
    final_labels
)


print("\n" + "=" * 70)
print("FINAL K-MEANS MODEL")
print("=" * 70)

print(
    "Selected K:",
    best_k
)

print(
    "Final inertia:",
    round(
        final_model.inertia_,
        2
    )
)

print(
    "Final Dunn Index:",
    round(
        final_dunn,
        6
    )
)


# Save clustering outputs so the results can be reviewed without rerunning the analysis
clustered_df.to_csv(
    "wholesale_customers_clustered.csv",
    index=False
)

metrics_df.to_csv(
    "kmeans_metrics.csv",
    index=False
)

centers_df.to_csv(
    "cluster_centers.csv"
)

prediction_table.to_csv(
    "first_10_customer_predictions.csv",
    index=False
)


print("\nSaved files:")

print(
    "wholesale_customers_clustered.csv"
)

print(
    "kmeans_metrics.csv"
)

print(
    "cluster_centers.csv"
)

print(
    "first_10_customer_predictions.csv"
)


print("\n" + "=" * 70)
print("OPTIONAL: HIERARCHICAL CLUSTERING")
print("=" * 70)

# Hierarchical clustering provides an alternative structure for comparison with K-Means
linkage_matrix = linkage(
    X_scaled,
    method="ward"
)


plt.figure(
    figsize=(14, 7)
)

dendrogram(
    linkage_matrix,
    truncate_mode="level",
    p=5
)

plt.xlabel(
    "Customers / Groups"
)

plt.ylabel(
    "Distance"
)

plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.tight_layout()

plt.show()


hierarchical_model = (
    AgglomerativeClustering(
        n_clusters=best_k,
        linkage="ward"
    )
)

hierarchical_labels = (
    hierarchical_model.fit_predict(
        X_scaled
    )
)

hierarchical_dunn = dunn_index(
    X_scaled,
    hierarchical_labels
)

print(
    "\nHierarchical Dunn Index:",
    round(
        hierarchical_dunn,
        6
    )
)


plt.figure(
    figsize=(10, 7)
)

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=hierarchical_labels,
    cmap="viridis",
    alpha=0.75
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    f"Hierarchical Customer Clustering (K={best_k})"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(True)

plt.tight_layout()

plt.show()


comparison = pd.DataFrame({

    "Method": [
        "K-Means",
        "Hierarchical"
    ],

    "Dunn Index": [
        final_dunn,
        hierarchical_dunn
    ]

})


print("\n" + "=" * 70)
print("K-MEANS VS HIERARCHICAL CLUSTERING")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# Compare both methods using the same Dunn Index definition
if final_dunn > hierarchical_dunn:

    print(
        "\nK-Means has the higher Dunn Index."
    )

elif hierarchical_dunn > final_dunn:

    print(
        "\nHierarchical clustering has "
        "the higher Dunn Index."
    )

else:

    print(
        "\nBoth methods have the same Dunn Index."
    )


print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)