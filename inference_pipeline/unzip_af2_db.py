import os
import tarfile
import json
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Setup argument parsing
parser = argparse.ArgumentParser(description='Unzip a specified number of .tar files into separate directories, log newly created directories with a progress bar, and utilize multi-processing for efficiency.')
parser.add_argument('num_files_to_unzip', type=int, help='The number of .tar files to unzip')
parser.add_argument('--directory', type=str, default="/media/af_database", help='Directory containing the .tar files (default is /media/af_database)')
args = parser.parse_args()

# Use arguments
directory = args.directory
num_files_to_unzip = args.num_files_to_unzip

# Record and log files placed in the current working directory
record_file_path = os.path.join(os.getcwd(), "af2_unzipped_record.json")
log_file_path = os.path.join(os.getcwd(), "af2_created_directories.log")

def load_record(record_file):
    """Load the record file containing the list of unzipped files and directories."""
    if os.path.exists(record_file):
        with open(record_file, "r") as file:
            return json.load(file)
    else:
        return {"unzipped_files": [], "created_directories": []}

def save_record(record_file, record_data):
    """Save the updated record to the file."""
    with open(record_file, "w") as file:
        json.dump(record_data, file)

def log_new_directories(log_file, new_directories):
    """Append newly created directories to the log file."""
    with open(log_file, "w") as file:
        for directory in new_directories:
            file.write(f"{directory}\n")

def unzip_file(file_info):
    """Function to unzip a single .tar file and return file info if new."""
    directory, f = file_info
    tar_path = os.path.join(directory, f)
    extract_dir = os.path.join(directory, f.replace(".tar", ""))

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        # Extract the .tar file
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir)
        return f, extract_dir, True  # Return file name, directory name, and True for newly created
    return f, extract_dir, False  # Return file name, directory name, and False, not newly created

def main(directory, num_files_to_unzip):
    """Main function to manage multiprocessing, progress bar, and record keeping."""
    record_data = load_record(record_file_path)
    unzipped_files = set(record_data.get("unzipped_files", []))
    created_directories = set(record_data.get("created_directories", []))
    new_directories_this_run = []

    # Find .tar files that haven't been unzipped yet
    tar_files = [f for f in os.listdir(directory) if f.endswith(".tar") and f not in unzipped_files]
    files_to_process = tar_files[:num_files_to_unzip]

    with Pool(processes=cpu_count()) as pool:
        for f, dir_name, is_new in tqdm(pool.imap_unordered(unzip_file, [(directory, f) for f in files_to_process]), total=len(files_to_process), desc="Unzipping files"):
            if is_new:
                new_directories_this_run.append(dir_name)
            unzipped_files.add(f)  # Keep the .tar extension for accurate tracking

    # Update and save the record of unzipped files and created directories
    record_data["unzipped_files"] = list(unzipped_files)
    record_data["created_directories"] = list(created_directories.union(new_directories_this_run))
    save_record(record_file_path, record_data)

    # Log newly created directories, if any
    if new_directories_this_run:
        log_new_directories(log_file_path, new_directories_this_run)
        print("Logged newly created directories.")

    if not files_to_process:
        print("No more files to unzip or specified number of files exceeds available unzipped files.")

if __name__ == "__main__":
    main(directory, num_files_to_unzip)
