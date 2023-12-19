from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, BertConfig


# Load the SST-2 dataset from the Hugging Face dataset library
dataset = load_dataset('glue', 'sst2')

# Load the tokenizer for BERT LSH
tokenizer = AutoTokenizer.from_pretrained('./bertlsh')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Load your pre-trained BertForMaskedLM
masked_lm_model = BertForMaskedLM.from_pretrained('path_to_your_model')

# Create a new model for sequence classification using the same configuration
config = masked_lm_model.config
model = BertForSequenceClassification(config)