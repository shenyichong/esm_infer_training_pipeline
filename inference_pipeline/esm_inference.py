import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import uuid 
import argparse


def main(input_csv, output_csv, model_path):
    # model_path = "/root/cell/DimerPLM/esm_training_example_code/esm2_t33_650M_UR50D-feb25/checkpoint-15830/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # df = pd.read_csv("/data_0/home/shenyichong/extract_seq_from_cifs/subfolder_1_sequences.csv", header=None)
    # for i in range(2, 12):
    #     new_df = pd.read_csv(f"/data_0/home/shenyichong/extract_seq_from_cifs/subfolder_{i}_sequences.csv", header=None)
    #     df = pd.concat([df, new_df], ignore_index=True)
    # df[1] = df[1].str.strip(to_strip =";")
    # df.to_csv("af_seq.csv")

    df = pd.read_csv(input_csv, names=["file_path", "amino_acid_sequence"], header=0) 

    test_sequences = df["amino_acid_sequence"].astype(str).tolist()
    test_pdb_names = df["file_path"].astype(str).tolist()
    test_inputs = tokenizer(test_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    test_dataset = Dataset.from_dict(test_inputs)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Custom collate_fn to handle conversion from lists to tensors
    def collate_fn(batch):
        # Convert list of dicts to dict of lists
        batch = {key: [d[key] for d in batch] for key in batch[0]}

        # Convert lists to tensors
        input_ids = torch.tensor(batch['input_ids'])
        attention_mask = torch.tensor(batch['attention_mask'])
        
        # Optionally handle labels if you have them
        if 'labels' in batch:
            labels = torch.tensor(batch['labels'])
            return input_ids, attention_mask, labels

        return input_ids, attention_mask

    # Create DataLoader with custom collate_fn
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Assuming you have already set your model to evaluation mode using model.eval()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the selected device
    model = model.to(device)

    # Wrap the model in DataParallel to utilize all available GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Initialize a list to store the predictions
    predictions = []

    # Initialize a list to store the predicted class labels (if needed)
    predicted_labels = []

    # Using DataLoader, this loop should iterate over batches
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids, attention_mask = batch[:2]  # If you have labels, they would be in batch[2]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass, get model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Apply softmax to the logits to get probabilities
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert the probabilities to CPU and to a NumPy array
            predicted_probs = scores.cpu().numpy()

            # Extend the predictions list with the probabilities
            predictions.extend(predicted_probs)

            # Optionally, if you want to get the predicted class labels
            predicted_labels.extend(torch.argmax(scores, dim=-1).cpu().numpy())

    # Debug: Check the total number of predictions
    print(f"Total predictions: {len(predictions)}")

    # Each element of predictions is now a list of probabilities for the three classes
    # For example: [0.2, 0.3, 0.5] for a single data point

    # Debug: Check if the number of predictions matches the expected number of data points
    assert len(predictions) == len(test_loader.dataset), "Number of predictions does not match number of data points."

    # Convert predictions and predicted_labels to NumPy arrays
    predictions_np = np.array(predictions)
    predicted_labels_np = np.array(predicted_labels)

    # add unique identifier
    unique_identifiers = [str(uuid.uuid4()) for _ in range(len(test_sequences))]

    # Add input sequences, predictions, and predicted labels to a DataFrame
    results_df = pd.DataFrame({
        'Unique Identifier': unique_identifiers,
        'PDB FILE NAME': test_pdb_names[:len(predictions)],
        'Input Sequence': test_sequences[:len(predictions)],
        'Prediction Class 0': predictions_np[:, 0],
        'Prediction Class 1': predictions_np[:, 1],
        'Prediction Class 2': predictions_np[:, 2],
        'Predicted Label': predicted_labels_np
    })

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform ESM-based classification model inference on sequences.')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file path containing sequences.')
    parser.add_argument('--output_csv', type=str, required=True, help='Output directory for results CSV file.')
    parser.add_argument('--model_path', type=str, required=True, help='esm model path.')
    args = parser.parse_args()

    main(args.input_csv, args.output_csv, args.model_path)