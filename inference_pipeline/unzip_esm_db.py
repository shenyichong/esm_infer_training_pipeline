import argparse
import tarfile
import os
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def load_unzipped_record(record_file):
    """Load the cumulative record of unzipped files."""
    if os.path.exists(record_file):
        with open(record_file, 'r') as file:
            return json.load(file)
    else:
        return {"unzipped_files": []}

def save_unzipped_record(record_file, record_data):
    """Save the updated cumulative record."""
    with open(record_file, 'w') as file:
        json.dump(record_data, file)

def unzip_file(task):
    tar_gz_file, input_dir, output_dir = task
    try:
        with tarfile.open(os.path.join(input_dir, tar_gz_file), "r:gz") as tar:
            tar.extractall(path=output_dir)
        return tar_gz_file[:-7], tar_gz_file  # Return folder name and full file name
    except Exception as e:
        print(f"Failed to unzip {tar_gz_file}: {str(e)}")
        return None, None

def unzip_files(num_files, input_dir, record_file):
    record_data = load_unzipped_record(record_file)
    unzipped_files = record_data["unzipped_files"]
    tasks = []
    task_cnt = 0 
    for tar_gz_file in os.listdir(input_dir):
        if tar_gz_file.endswith(".tar.gz") and tar_gz_file not in unzipped_files:
            output_dir = os.path.join(input_dir, tar_gz_file[:-7])  # Remove .tar.gz extension for folder name
            if task_cnt < num_files:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                tasks.append((tar_gz_file, input_dir, output_dir))
                task_cnt += 1
            else:
                break

    newly_unzipped = []
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(unzip_file, tasks), total=len(tasks), desc="Unzipping files"):
            folder_name, full_file_name = result
            if folder_name and full_file_name:
                newly_unzipped.append(folder_name)
                unzipped_files.append(full_file_name)

    # Save the updated cumulative record
    record_data["unzipped_files"] = unzipped_files
    save_unzipped_record(record_file, record_data)

    # Log newly unzipped directories for the current run
    with open("esm_created_directories.log", 'w') as log_file:
        for folder_name in newly_unzipped:
            log_file.write(f"{folder_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unzip a specified number of .tar.gz files from a directory, avoiding duplication.')
    parser.add_argument('--num_files', type=int, required=True, help='Number of .tar.gz files to unzip')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the .tar.gz files')
    args = parser.parse_args()

    # Cumulative record file path
    record_file = "esm_unzipped_record.json"

    unzip_files(args.num_files, args.input_dir, record_file)
