import json
import os


# import torch
import pandas as pd

from datasets import Dataset, DatasetDict
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def read_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                print("Error: File contains invalid JSON.")
                return None
    else:
        print(f"Error: File {file_path} does not exist.")
        return None

def print_json(data):
    if data is not None:
        print(json.dumps(data, indent=4))
    else:
        print("No data to print.")

# Example usage
file_path = 'user_employee_feedbacks/user_employee_feedbacks.json'
data = read_json(file_path)
print_json(data['user_responses'][-1])


# Create a Dataset from the JSON data
dataset = Dataset.from_list(data['user_responses'])

# Create a DatasetDict with the dataset as the test set
dataset_dict = DatasetDict({
    'test': dataset
})

# Print the DatasetDict to check its structure
print(dataset_dict)


model = 'habibul08/employee-sentiment-tracker'


# Load tokenizer and model
# tokenizer = BertTokenizer.from_pretrained(model)
# model = BertForSequenceClassification.from_pretrained(model, num_labels=2)  # 2 classes: negative and positive
# test_encodings = tokenizer(dataset_dict, padding=True, truncation=True, max_length=128)
