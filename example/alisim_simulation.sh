#!/bin/bash

lengths=(100 1000 10000 100000 1000000)
iqtree_executable_path="/Users/u7826985/Projects/GPU-utest/iqtree3/cmake-build-vanila/iqtree3"
tree_file="tree.nwk"
DATASET_DIR="test_data/"

# Loop through each taxa folder
for TAXA_DIR in "$DATASET_DIR"{4,100,500,1000,10000}taxa; do
    echo "Processing folder: $TAXA_DIR"

    echo "Current directory: $(pwd)"

    cd "$TAXA_DIR" || { echo "Failed to change directory to $TAXA_DIR"; exit 1; }

    if [ -f "$tree_file" ]; then
        echo "Current directory: $(pwd)"

        for length in "${lengths[@]}"; do
            echo "Running simulation for length: $length"
            $iqtree_executable_path --alisim alignment_${length} -m JC -t $tree_file --length ${length}
            if [ $? -ne 0 ]; then
                echo "Simulation failed for length: $length"
                exit 1
            fi

        done

    else
        echo "File not found: $TAXA_DIR/$tree_file"
    fi

    cd - || { echo "Failed to return to previous directory"; exit 1; }

    echo "--------------------------------------"

done
