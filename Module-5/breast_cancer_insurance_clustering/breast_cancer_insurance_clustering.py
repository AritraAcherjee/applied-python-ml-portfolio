import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN

# Load the two datasets
# The breast cancer file is the refined dataset created earlier
breast = pd.read_csv("data_refined.csv")

# Insurance data is loaded separately for its own clustering analysis
insurance = pd.read_csv("insurance.csv")

print("Breast Cancer shape:", breast.shape)
print("Insurance shape:", insurance.shape)

# Prepare the data for clustering
# ID and diagnosis columns are removed because they should not influence distance-based clusters
breast = breast.drop(
    columns=["id", "Unnamed: 32", "diagnosis"],
    errors="ignore"
)

# Median filling keeps numeric columns usable without being overly affected by extreme values
breast = breast.fillna(breast.median(numeric_only=True))

# Use the most common value for categories and the median for numerical fields
for col in insurance.columns:
    if insurance[col].dtype == "object":
        insurance[col] = insurance[col].fillna(insurance[col].mode()[0])
    else:
        insurance[col] = insurance[col].fillna(insurance[col].median())

# Clustering requires numerical input, so categorical values are converted to dummy variables
insurance = pd.get_dummies(
    insurance,
    columns=["sex", "smoker", "region"],
    drop_first=True
)

# Scale both datasets so features with larger numeric ranges do not dominate the clusters
scaler_breast = StandardScaler()
scaler_insurance = StandardScaler()

X_breast = scaler_breast.fit_transform(breast)
X_insurance = scaler_insurance.fit_transform(insurance)

# Elbow method
def elbow_method(X, title):
    inertia = []

    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertia.append(model.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 11), inertia, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title(title)
    plt.show()


elbow_method(X_breast, "Breast Cancer - Elbow Method")
elbow_method(X_insurance, "Insurance - Elbow Method")

# K-Means clustering
# These cluster counts can be adjusted if the elbow plots suggest better values
breast_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
insurance_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

breast["KMeans_Cluster"] = breast_kmeans.fit_predict(X_breast)
insurance["KMeans_Cluster"] = insurance_kmeans.fit_predict(X_insurance)

print("\nBreast Cancer K-Means clusters:")
print(breast["KMeans_Cluster"].value_counts())

print("\nInsurance K-Means clusters:")
print(insurance["KMeans_Cluster"].value_counts())

# Mean Shift clustering
# Estimate bandwidth from the data instead of choosing one value arbitrarily
breast_bandwidth = estimate_bandwidth(
    X_breast,
    quantile=0.2,
    n_samples=min(500, len(X_breast))
)

insurance_bandwidth = estimate_bandwidth(
    X_insurance,
    quantile=0.2,
    n_samples=min(500, len(X_insurance))
)

print("\nEstimated Breast Cancer bandwidth:", breast_bandwidth)
print("Estimated Insurance bandwidth:", insurance_bandwidth)

breast_mean_shift = MeanShift(
    bandwidth=breast_bandwidth,
    bin_seeding=True
)

insurance_mean_shift = MeanShift(
    bandwidth=insurance_bandwidth,
    bin_seeding=True
)

breast["MeanShift_Cluster"] = breast_mean_shift.fit_predict(X_breast)
insurance["MeanShift_Cluster"] = insurance_mean_shift.fit_predict(X_insurance)

print("\nBreast Cancer Mean Shift clusters:")
print(breast["MeanShift_Cluster"].value_counts())

print("\nInsurance Mean Shift clusters:")
print(insurance["MeanShift_Cluster"].value_counts())

# Compare several Mean Shift bandwidth values to see how cluster counts change
print("\nMean Shift bandwidth test - Breast Cancer")

for bandwidth in [2, 3, 4, 5]:
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = model.fit_predict(X_breast)

    print(
        "Bandwidth:",
        bandwidth,
        "| Number of clusters:",
        len(set(labels))
    )

# DBSCAN clustering
# This method can identify dense groups while separating unusual points as noise
dbscan = DBSCAN(
    eps=2.5,
    min_samples=5
)

breast["DBSCAN_Cluster"] = dbscan.fit_predict(X_breast)

print("\nBreast Cancer DBSCAN clusters:")
print(breast["DBSCAN_Cluster"].value_counts())

print(
    "Number of DBSCAN clusters:",
    len(set(breast["DBSCAN_Cluster"])) -
    (1 if -1 in breast["DBSCAN_Cluster"].values else 0)
)

# In DBSCAN, a label of -1 means the observation was treated as noise or an outlier

# Save the cluster labels so the results can be reviewed later
breast.to_csv("breast_cancer_clustered.csv", index=False)
insurance.to_csv("insurance_clustered.csv", index=False)

print("\nFinished. Clustered datasets have been saved.")