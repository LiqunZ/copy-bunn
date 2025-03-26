#!/bin/bash

# Specify the folder containing the .job scripts
job_folder="bunn_search"

# Check if the folder exists
if [ -d "$job_folder" ]; then
    # Loop over all .job files in the folder
    for job_script in "$job_folder"/*.job; do
        if [ -f "$job_script" ]; then
            # Run the .job script using sbatch
            sbatch "$job_script"
        fi
    done
else
    echo "Folder not found: $job_folder"
fi