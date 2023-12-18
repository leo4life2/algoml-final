import os
from huggingface_hub import login, HfApi
from datasets import load_dataset

hf_token = os.environ.get("HF_TOKEN", None)
if not hf_token:
    raise Exception("No HF Token")
login(token=hf_token)
user_id = HfApi().whoami()["name"]
print(f"Welcome, '{user_id}' \n")

# Load the dataset
full_dataset = load_dataset("bookcorpus")

# Calculate the number of rows for 10%
num_rows = len(full_dataset['train'])
rows_10p = int(num_rows * 0.1)

# Calculate the starting index for the bottom 10%
start_index = int(num_rows * 0.9)

# Take the bottom 10% of rows
subset_dataset = full_dataset['train'].select(range(start_index, num_rows))
# Save to disk
subset_dataset.save_to_disk("algoml_bookcorpus_bottom_10p")
dataset_repo_id = f"{user_id}/algoml_bookcorpus_bottom_10p"
subset_dataset.push_to_hub(dataset_repo_id)