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

# Calculate the number of rows for 1%
num_rows = len(full_dataset['train'])
rows_1p = int(num_rows * 0.01)

# Calculate the starting index for the 49th percent
start_index = int(num_rows * 0.49)

# Take the bottom 10% of rows
subset_dataset = full_dataset['train'].select(range(start_index, start_index + rows_1p))
# Save to disk
subset_dataset.save_to_disk("algoml_bookcorpus_49_50p")
dataset_repo_id = f"{user_id}/algoml_bookcorpus_49_50p"
subset_dataset.push_to_hub(dataset_repo_id)