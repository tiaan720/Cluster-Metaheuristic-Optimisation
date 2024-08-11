import pandas as pd

# Step 1: Load the CSV file
file_path = "cluster_data/combined_clustering_results.csv"
df = pd.read_csv(file_path)

# Step 1.1: Remove columns named "Convergence Data" and "Diversity Data"
df = df.drop(columns=["Convergence Data", "Diversity Data"])

# Step 2: Remove unnecessary columns except "Dataset"
df_cleaned = df.drop(columns=["Davies Bouldin Index", "Source"])

# Step 3: Split the dataframe into separate dataframes based on the "Dataset" column
datasets = df_cleaned["Dataset"].unique()
dataframes = {
    dataset: df_cleaned[df_cleaned["Dataset"] == dataset] for dataset in datasets
}

# Step 4: Remove the "Dataset" column and keep only one KMeans row per dataset, removing duplicates
filtered_dataframes = {}
for dataset, dataframe in dataframes.items():
    dataframe = dataframe.drop(columns=["Dataset"])
    kmeans_rows = dataframe[dataframe["Algorithm"] == "KMeans"].drop_duplicates(
        subset=["Algorithm"]
    )
    filtered_dataframes[dataset] = pd.concat(
        [kmeans_rows, dataframe[dataframe["Algorithm"] != "KMeans"]]
    )

# Step 5: Save the final dataframes into separate CSV files
final_output_files = {}
for dataset, dataframe in filtered_dataframes.items():
    output_file = f"cluster_data/final_{dataset}_clustering_results.csv"
    dataframe.to_csv(output_file, index=False)
    final_output_files[dataset] = output_file

# Output the final file paths
print(final_output_files)
