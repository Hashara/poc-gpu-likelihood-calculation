#!/bin/bash

lengths=(100 1000 10000 100000 1000000)
executable_type=("cpu" "eigen" "openacc")
tree_file="tree.nwk"
DATASET_DIR="test_data/"

# Loop through each taxa folder
for TAXA_DIR in "$DATASET_DIR"{4,100,500,1000,10000}taxa; do
    echo "Processing folder: $TAXA_DIR"
    taxa_size=$(basename "$TAXA_DIR")

    echo "Current directory: $(pwd)"

    cd "$TAXA_DIR" || { echo "Failed to change directory to $TAXA_DIR"; exit 1; }

    if [ -f "$tree_file" ]; then
        echo "Current directory: $(pwd)"

        for length in "${lengths[@]}"; do
            echo "Running likelihood for length: $length taxa: $taxa_size"

            #loop through each executable type
            for type in "${executable_type[@]}"; do
                executable_path="../../build/$type/gpulcal"
                echo "Using executable: $executable_path"
                
                if [ -f "$executable_path" ]; then
                    echo "Running test for length: $length with $type"
                    $executable_path -s alignment_100.phy -te tree.nwk -prefix output_${taxa_size}taxa_${length}_${type}.txt

                    if [ $? -ne 0 ]; then
                        echo "run failed for length: $length with $type for $taxa_size taxa"
                        exit 1
                    fi
                else
                    echo "Executable not found: $executable_path"
                fi

            done

        done

    else
        echo "File not found: $TAXA_DIR/$tree_file"
    fi

    cd - || { echo "Failed to return to previous directory"; exit 1; }

    echo "--------------------------------------"

done
