from Bio import PDB
import os
import csv
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from Bio.SeqUtils import seq3

def extract_amino_acids(pdb_file_path):
    """Extracts amino acid sequence from a .pdb file using Bio.PDB."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file_path, pdb_file_path)
    amino_acids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    amino_acids.append(PDB.Polypeptide.three_to_one(residue.get_resname()))
                    # try:
                    #     amino_acids.append(seq3(residue.get_resname()))
                    # except KeyError:
                    #     print(f"Unknown amino acid: {three_letter_code}")
                    #     amino_acids.append('X')  # 'X' for unknown amino acids
    return os.path.basename(pdb_file_path), ''.join(amino_acids)

def init_process(directory):
    """Initialize processing of PDB files and gather paths."""
    pdb_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(".pdb")]
    return pdb_files

def write_sequences_to_csv(results, output_csv):
    """Writes amino acid sequences and their file paths to a CSV."""
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['file_path', 'amino_acid_sequence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({'file_path': result[0], 'amino_acid_sequence': result[1]})

def process_files(directory, output_csv):
    pdb_files = init_process(directory)
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(extract_amino_acids, pdb_files), total=len(pdb_files)))
    write_sequences_to_csv(results, output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract sequences from PDB files.')
    parser.add_argument('directory', type=str, help='Directory containing the PDB files')
    parser.add_argument('output_csv', type=str, help='Output CSV file path')
    args = parser.parse_args()

    # Execute the function with multiprocessing and progress bar
    process_files(args.directory, args.output_csv)

    print(f"CSV file has been created at {args.output_csv}")
