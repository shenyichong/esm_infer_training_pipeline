#!/bin/bash

# Step 1: Unzip files
NUM_FILES_TO_UNZIP=$1
python unzip_af2_db.py $NUM_FILES_TO_UNZIP

# Step 2: Read newly created directories from af2_created_directories.log
NEW_DIRECTORIES=$(cat af2_created_directories.log)

# Base destination directory for all filtered PDB files
BASE_DESTINATION="plddt_70_af2_pdb"
THRESHOLD=70
NUM_PROCESSES=32

# Ensure the base destination directory exists
mkdir -p $BASE_DESTINATION

# Process each newly created directory
for FULL_DIR_PATH in $NEW_DIRECTORIES; do
    # Extract just the folder name from the full directory path
    FOLDER_NAME=$(basename $FULL_DIR_PATH)

    # Define the specific destination directory for the filtered PDB files
    DESTINATION_FOLDER="${BASE_DESTINATION}/${FOLDER_NAME}"

    # Ensure the specific destination directory exists
    mkdir -p $DESTINATION_FOLDER

    # Execute the filtering process for the current directory
    python filter_plddt.py $FULL_DIR_PATH --threshold $THRESHOLD --num_processes $NUM_PROCESSES --destination $DESTINATION_FOLDER

    echo "Filtered PDB files from $FULL_DIR_PATH are located in $DESTINATION_FOLDER"


done

# Step 3: Unzip all .pdb.gz files to .pdb within the filtered directories
for DIR in $NEW_DIRECTORIES; do
    FOLDER_NAME=$(basename $DIR)
    DIRECTORY="${BASE_DESTINATION}/${FOLDER_NAME}/"

    for FILE in "$DIRECTORY"*.pdb.gz; do
      # Skip loop if directory is empty
      [ -e "$FILE" ] || continue

      # Construct the output filename by removing the .gz extension
      OUTPUT="${FILE%.gz}"

      # Unzip the file
      gzip -d -c "$FILE" > "$OUTPUT"

      echo "Unzipped $FILE to $OUTPUT"
    done
done

echo "All .pdb.gz files have been uncompressed."

# # Step 4: Extract sequences from PDBs and output to CSV files
SEQS_DESTINATION="af2_plddt_70_seqs"
mkdir -p $SEQS_DESTINATION

for DIR in $NEW_DIRECTORIES; do
    FOLDER_NAME=$(basename $DIR)
    INPUT_DIR="${BASE_DESTINATION}/${FOLDER_NAME}/"
    OUTPUT_CSV="${SEQS_DESTINATION}/${FOLDER_NAME}.csv"

    echo "$DIR"
    echo "$OUTPUT_CSV"
    python extract_seq_from_pdbs.py "$INPUT_DIR" "$OUTPUT_CSV"

    echo "Extracted sequences from $INPUT_DIR to $OUTPUT_CSV"
done

echo "Sequence extraction completed."

# Step 5: Perform inference with the ESM-based classification model on sequences
INFERENCE_DESTINATION="af2_plddt_70_infer_result"
mkdir -p "$INFERENCE_DESTINATION"

MODEL_PATH="/root/cell/DimerPLM/esm_training_example_code/esm2_t33_650M_UR50D-feb25/checkpoint-15830/"

# Ensure af2_created_directories.log exists and is not empty
if [ ! -s af2_created_directories.log ]; then
    echo "af2_created_directories.log is missing or empty. Exiting..."
    exit 1
fi

while IFS= read -r FULL_DIR_PATH; do
    DIR_NAME=$(basename "$FULL_DIR_PATH")
    INPUT_CSV="${SEQS_DESTINATION}/${DIR_NAME}.csv"
    
    # Check if the input CSV file exists
    if [ -f "$INPUT_CSV" ]; then
        OUTPUT_CSV="${INFERENCE_DESTINATION}/${DIR_NAME}_results.csv"
        
        python esm_inference.py --input_csv "$INPUT_CSV" --output_csv "$OUTPUT_CSV" --model_path "$MODEL_PATH"
        
        echo "Inference results for $INPUT_CSV are saved to $OUTPUT_CSV"
    else
        echo "No sequence CSV file found for $DIR_NAME, skipping..."
    fi
done < af2_created_directories.log

echo "Inference process completed."

echo "Process completed."
