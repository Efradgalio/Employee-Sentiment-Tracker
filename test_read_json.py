import json
import os

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
