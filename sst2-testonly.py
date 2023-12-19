from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.models.bert.modeling_lsh_bert import BertLSHForSequenceClassification
import os
import json

# Load the SST-2 dataset from the Hugging Face dataset library
full_dataset = load_dataset('glue', 'sst2')

# Use the original validation set as the test set
test_dataset = full_dataset['validation']

# Load the tokenizer and model from the saved directory
tokenizer = AutoTokenizer.from_pretrained('./bertlsh')
model = BertLSHForSequenceClassification.from_pretrained('./bertlsh-sst2')

# Tokenize the test dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

# Tokenize the test set
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Define the output directory
output_dir = './bertlsh-sst2-test2hash4band'

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,  # Use the same output directory as before
    do_train=False,  # Disable training
    do_eval=True,  # Enable evaluation
    per_device_eval_batch_size=64,  # Adjust the batch size if necessary
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test,  # Only pass the test dataset for evaluation
)

# Evaluate the model on the test set
test_results = trainer.evaluate()

# Save the test results to the output directory
test_results_path = os.path.join(output_dir, 'test_results.json')
with open(test_results_path, 'w') as result_file:
    json.dump(test_results, result_file)

print("Evaluation on test set complete. Results saved to:", test_results_path)