#!/bin/bash

# Array of datasets
datasets=("students" "wine" "iris" "breast_cancer")

# Array of optimization algorithms
algorithms=("PSO" "GA" "SA" "TS")

# Loop through each dataset
for dataset in "${datasets[@]}"; do
  # Loop through each optimization algorithm
  for algorithm in "${algorithms[@]}"; do
    # Print a message indicating which combination is being run
    echo "Running $algorithm on $dataset dataset..."
    
    # Run the main.py script with the current dataset and optimization algorithm
    python src/main.py --dataset "$dataset" --optimization "$algorithm"
  done
done

echo "All combinations have been run."
