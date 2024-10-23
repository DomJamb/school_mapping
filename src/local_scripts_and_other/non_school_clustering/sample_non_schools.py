import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_euclidean_distance(point_df, point):
    # Transform geometry string to latitude, longitude tuple
    point_df = point_df.strip().replace("POINT (", "").replace(")", "")
    point_df = tuple(map(float, point_df.split()))

    # Calculate Euclidean distance
    dist =  ((point_df[0] - point[0]) ** 2 + (point_df[1] - point[1]) ** 2) ** 0.5

    return dist

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.realpath(__file__))

    f = os.path.join(cwd, "anditi_school.csv")
    dataset = pd.read_csv(f)
    
    assert len(dataset) == 1003

    with open(os.path.join(cwd, "anditi_cluster_1.txt"), "r") as f:
        cluster_1 = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    
    with open(os.path.join(cwd, "anditi_cluster_2.txt"), "r") as f:
        cluster_2 = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

    # Filter dataframe for each cluster
    cluster_1_rows = dataset.loc[dataset["image"].isin([f"{img_id}.jpeg" for img_id in cluster_1])]
    cluster_2_rows = dataset.loc[dataset["image"].isin([f"{img_id}.jpeg" for img_id in cluster_2])]

    assert len(cluster_1_rows) + len(cluster_2_rows) == 1003

    # Calculate centroids
    centroid1 = (cluster_1_rows["lon"].mean(), cluster_1_rows["lat"].mean())
    centroid2 = (cluster_2_rows["lon"].mean(), cluster_2_rows["lat"].mean())

    print(f"Centroid 1 (lon, lat): {centroid1}")
    print(f"Centroid 2 (lon, lat): {centroid2}")

    # Load non schools dataset
    f = os.path.join(cwd, "non_schools.csv")
    dataset_ns = pd.read_csv(f)

    ### CENTROID 1
    # Calculate distance from centroid 1
    dataset_ns["c1_dist"] = dataset_ns["geometry"].apply(lambda row: calculate_euclidean_distance(row, centroid1))

    # Calculate probability to belong to centroid 1 (1 / d1)
    dataset_ns["c1_prob"] = 1 / (dataset_ns["c1_dist"] + 1e-10)
    dataset_ns["c1_prob"] = dataset_ns["c1_prob"] / dataset_ns["c1_prob"].sum()

    # Plot sampling probability
    plt.figure(figsize=(8, 6))
    plt.title("Sampling Probability Depending on Distance to North Centroid")
    plt.xlabel("Distance")
    plt.ylabel("Probability")
    plt.scatter(dataset_ns["c1_dist"], dataset_ns["c1_prob"])
    plt.savefig(f"probability_plot_c1.png")

    # Choose nonschools for cluster 1 based on probability 1
    cluster1_ns_indices = np.random.choice(dataset_ns.index, size=len(cluster_1_rows), replace=False, p=dataset_ns["c1_prob"])
    cluster1_ns = dataset_ns.loc[cluster1_ns_indices]

    # Remove sampled nonschools
    dataset_ns = dataset_ns.drop(cluster1_ns_indices)

    ### CENTROID 2
    # Calculate distance from centroid 2
    dataset_ns["c2_dist"] = dataset_ns["geometry"].apply(lambda row: calculate_euclidean_distance(row, centroid2))

    # Calculate probability to belong to centroid 2 (1 / d2)
    dataset_ns["c2_prob"] = 1 / (dataset_ns["c2_dist"] + 1e-10)
    dataset_ns["c2_prob"] = dataset_ns["c2_prob"] / dataset_ns["c2_prob"].sum()

    # Plot sampling probability
    plt.figure(figsize=(8, 6))
    plt.title("Sampling Probability Depending on Distance to South Centroid")
    plt.xlabel("Distance")
    plt.ylabel("Probability")
    plt.scatter(dataset_ns["c2_dist"], dataset_ns["c2_prob"])
    plt.savefig(f"probability_plot_c2.png")

    # Choose nonschools for cluster 2 based on probability 2
    cluster2_ns_indices = np.random.choice(dataset_ns.index, size=len(cluster_2_rows), replace=False, p=dataset_ns["c2_prob"])
    cluster2_ns = dataset_ns.loc[cluster2_ns_indices]

    # Drop columns related to centroid 1
    cluster2_ns.drop(columns=["c1_dist", "c1_prob"], inplace=True)

    # Save selected non schools for each cluster
    cluster1_ns.to_csv(os.path.join(cwd, "non_schools_cluster_1.csv"), index=False)
    cluster2_ns.to_csv(os.path.join(cwd, "non_schools_cluster_2.csv"), index=False)