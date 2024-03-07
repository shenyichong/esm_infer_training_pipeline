import os
import gzip
import shutil
import argparse
from multiprocessing import Pool
from tqdm import tqdm

def extract_plddt_scores(pdb_gz_file):
    plddt_scores = []
    try:
        with gzip.open(pdb_gz_file, 'rt', encoding='ISO-8859-1') as file:
            for line in file:
                if line.startswith('ATOM'):
                    columns = line.split()
                    try:
                        plddt_score = float(columns[-2])
                        plddt_scores.append(plddt_score)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error processing file {pdb_gz_file}: {e}")
        return []
    return plddt_scores

def calculate_overall_plddt(plddt_scores):
    return sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0

def get_pdb_files(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdb.gz')]

def process_pdb_file(args):
    pdb_file, threshold = args
    plddt_scores = extract_plddt_scores(pdb_file)
    overall_plddt = calculate_overall_plddt(plddt_scores)
    if overall_plddt >= threshold:
        return pdb_file
    return None

def process_folders_parallel(folders, threshold, num_processes):
    with Pool(processes=num_processes) as pool:
        results = []
        for folder_path in folders:
            if os.path.isdir(folder_path):
                pdb_files = get_pdb_files(folder_path)
                tasks = [(pdb_file, threshold) for pdb_file in pdb_files]
                for _ in tqdm(pool.imap_unordered(process_pdb_file, tasks), total=len(pdb_files), desc=f'Processing {folder_path}'):
                    results.append(_)
                results = [file for file in results if file is not None]
    return results

def copy_files_to_folder(files, destination_folder):
    for file in files:
        try:
            shutil.copy2(file, destination_folder)
            print(f"Copied {file} to {destination_folder}")
        except Exception as e:
            print(f"Error copying {file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process some folders.')
    parser.add_argument('folders', metavar='folder', type=str, nargs='+', help='a folder to process')
    parser.add_argument('--threshold', type=int, default=70, help='PLDDT score threshold')
    parser.add_argument('--num_processes', type=int, default=os.cpu_count(), help='Number of processes to use')
    parser.add_argument('--destination', type=str, required=True, help='Destination folder for files meeting the threshold')
    args = parser.parse_args()

    filtered_files = process_folders_parallel(args.folders, args.threshold, args.num_processes)
    copy_files_to_folder(filtered_files, args.destination)

if __name__ == "__main__":
    main()
