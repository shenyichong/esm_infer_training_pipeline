#!/bin/bash

# Input number of .tar.gz files to unzip
NUM_FILES_TO_UNZIP=$1
ESM_DATABASE_DIR="/data_0/esm_altas_data"  # Directory containing .tar.gz files
UNZIP_DESTINATION="/data_0/esm_altas_data" # Directory containing unzip directory
SEQS_DESTINATION="/home/shenyichong/pdb_inference_pipline/esm_plddt_60_seqs"  # Where to save sequence CSV files
INFER_RESULTS_DESTINATION="/home/shenyichong/pdb_inference_pipline/esm_plddt_60_infer_result"  # Where to save inference results
MODEL_PATH="/root/cell/DimerPLM/esm_training_example_code/esm2_t33_650M_UR50D-feb25/checkpoint-15830/"  # ESM model path 

# # Step 1: Unzip .tar.gz files
# python unzip_esm_db.py --num_files $NUM_FILES_TO_UNZIP --input_dir "$ESM_DATABASE_DIR" 

# Newly unzipped directories will be logged in esm_created_directories.log
NEW_DIRECTORIES=$(cat esm_created_directories.log)

# Step 2: Extract sequences from PDBs
mkdir -p $SEQS_DESTINATION

for DIR in $NEW_DIRECTORIES; do
    FOLDER_NAME=$(basename $DIR)
    INPUT_DIR="${UNZIP_DESTINATION}/${FOLDER_NAME}/"
    OUTPUT_CSV="${SEQS_DESTINATION}/${FOLDER_NAME}.csv"

    python extract_seq_from_pdbs.py "$INPUT_DIR" "$OUTPUT_CSV"

    echo "Extracted sequences from $INPUT_DIR to $OUTPUT_CSV"
done

# Step 3: Inference using ESM-based classification model
mkdir -p $INFER_RESULTS_DESTINATION

# Ensure created_directories.log exists and is not empty
if [ ! -s esm_created_directories.log ]; then
    echo "esm_created_directories.log is missing or empty. Exiting..."
    exit 1
fi

while IFS= read -r FULL_DIR_PATH; do
    DIR_NAME=$(basename "$FULL_DIR_PATH")
    INPUT_CSV="${SEQS_DESTINATION}/${DIR_NAME}.csv"
    
    # Check if the input CSV file exists
    if [ -f "$INPUT_CSV" ]; then
        OUTPUT_CSV="${INFER_RESULTS_DESTINATION}/${DIR_NAME}_results.csv"
        
        python esm_inference.py --input_csv "$INPUT_CSV" --output_csv "$OUTPUT_CSV" --model_path "$MODEL_PATH"
        
        echo "Inference results for $INPUT_CSV are saved to $OUTPUT_CSV"
    else
        echo "No sequence CSV file found for $DIR_NAME, skipping..."
    fi
done < esm_created_directories.log

echo "Inference process completed."

echo "ESM database processing completed."
