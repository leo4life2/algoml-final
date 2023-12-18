import os
import torch
import multiprocessing

from transformers import AutoModel, BertTokenizerFast, AutoTokenizer, BertConfig, BertModel, AdamW
from transformers.models.bert.modeling_lsh_bert import BertLSHModel
from datasets import concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import login, HfApi
from tqdm import tqdm
from itertools import chain
from torch.utils.data import DataLoader

TRAIN_TOKENIZER = False
PREPROCESS_DATASET = False

# Huggingface login
hf_token = os.environ.get("HF_TOKEN", None)
if not hf_token:
    raise Exception("No HF Token")
login(token=hf_token)
user_id = HfApi().whoami()["name"]
print(f"Welcome, '{user_id}' \n")

# # Dataset
# bookcorpus = load_dataset("bookcorpus", split="train")

# # Training a tokenizer
# # repository id for the tokenizer
# tokenizer_id="algoml-bert-tokenizer"
# if TRAIN_TOKENIZER:

#     # create a python generator to dynamically load the data
#     def batch_iterator(batch_size=10000):
#         for i in tqdm(range(0, len(bookcorpus), batch_size)):
#             yield bookcorpus[i : i + batch_size]["text"]

#     # create a tokenizer from existing one to re-use special tokens
#     tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

#     bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
#     bert_tokenizer.save_pretrained("tokenizer")
    
#     bert_tokenizer.push_to_hub(f"{user_id}/{tokenizer_id}")
# else:
#     if os.path.isdir(os.path.join(os.getcwd(), "tokenizer")):
#         print("Loading local tokenizer")
#         bert_tokenizer = AutoTokenizer.from_pretrained(f"tokenizer")
#     else:
#         print("Can't find tokenizer locally, loading from HF")
#         bert_tokenizer = AutoTokenizer.from_pretrained(f"{user_id}/{tokenizer_id}")
#         bert_tokenizer.save_pretrained("tokenizer")
    
    
# # Preprocess dataset
# if PREPROCESS_DATASET:

#     # load tokenizer
#     # tokenizer = AutoTokenizer.from_pretrained(f"{user_id}/{tokenizer_id}")
#     tokenizer = AutoTokenizer.from_pretrained("tokenizer")
#     num_proc = multiprocessing.cpu_count()
#     print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

#     def group_texts(examples):
#         tokenized_inputs = tokenizer(
#         examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
#         )
#         return tokenized_inputs

#     tokenized_datasets = bookcorpus.map(group_texts, batched=True, remove_columns=["text"], num_proc=num_proc)
#     print(tokenized_datasets.features)

#     # Main data processing function that will concatenate all texts from our dataset and generate chunks of
#     # max_seq_length.
#     def group_texts(examples):
#         # Concatenate all texts.
#         concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#         total_length = len(concatenated_examples[list(examples.keys())[0]])
#         # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#         # customize this part to your needs.
#         if total_length >= tokenizer.model_max_length:
#             total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
#         # Split by chunks of max_len.
#         result = {
#             k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
#             for k, t in concatenated_examples.items()
#         }
#         return result

#     tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
#     # shuffle dataset
#     tokenized_datasets = tokenized_datasets.shuffle(seed=34)

#     print(f"the dataset contains in total {len(tokenized_datasets)*tokenizer.model_max_length} tokens")
#     # the dataset contains in total 3417216000 tokens

#     # push dataset to hugging face
#     tokenized_datasets.push_to_hub(f"{user_id}/processed_bert_dataset")
# else:
#     if os.path.isdir(os.path.join(os.getcwd(), "algoml_bookcorpus.hf")):
#         print("Loading local dataset")
#         tokenized_datasets = load_from_disk("algoml_bookcorpus.hf")
#     else:
#         print("Can't find dataset locally, loading from HF")
#         tokenized_datasets = load_dataset(f"{user_id}/processed_bert_dataset")
#         tokenized_datasets.save_to_disk("algoml_bookcorpus.hf")

config = BertConfig(
    vocab_size=32_000,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512,
    bands=8,
    num_hashes=4
)

model = BertLSHModel(config)
# model = BertModel(config)

# Generate some random data to feed the model
# Let's assume a batch size of 1 and a sequence length of 10 for this example
batch_size = 1
sequence_length = 10
random_input = torch.randint(config.vocab_size, (batch_size, sequence_length))

# Attention mask (assuming all tokens are not padding)
attention_mask = torch.ones(batch_size, sequence_length)

# Run the model
with torch.no_grad():  # Ensure no gradients are calculated
    output = model(random_input, attention_mask=attention_mask)

# Inspect the output
print(output)