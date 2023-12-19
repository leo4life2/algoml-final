from datasets import load_dataset
from transformers.models.bert.modeling_lsh_bert import BertLSHForQuestionAnswering
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer, Trainer, TrainingArguments
import os
import json

BENCH_LSH = False

full_dataset = load_dataset('squad')

# Splitting 5% of the training data for validation
train_testvalid = full_dataset['train'].train_test_split(test_size=0.05)
train_dataset = train_testvalid['train']
validation_dataset = train_testvalid['test']
test_dataset = full_dataset['validation']

# Load the tokenizer for BERT LSH
if BENCH_LSH:
    tokenizer = AutoTokenizer.from_pretrained('./bertlsh')
else:
    tokenizer = AutoTokenizer.from_pretrained('./origbert')

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the questions and contexts with truncation and padding
    tokenized_examples = tokenizer(
        examples['question'], examples['context'], 
        padding='max_length', truncation=True, max_length=384
    )
    
    # Prepare the list for start_positions and end_positions
    start_positions = []
    end_positions = []
    
    for i, (context, answer) in enumerate(zip(examples['context'], examples['answers'])):
        # Find the start and end token index for the answer in the context
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        
        # Find the token index corresponding to the start character of the answer
        token_start_index = tokenized_examples.char_to_token(i, start_char)
        token_end_index = tokenized_examples.char_to_token(i, end_char - 1)
        
        # If the answer is not within the tokens (due to truncation), set the cls index as answer
        if token_start_index is None:
            token_start_index = tokenizer.cls_token_id
        if token_end_index is None:
            token_end_index = tokenizer.cls_token_id
        
        # If we could not map the answer's end position in the text to a token in the
        # tokenized text, then the answer has been truncated.
        if token_end_index is not None:
            token_end_index += 1
        
        start_positions.append(token_start_index)
        end_positions.append(token_end_index)
    
    # Update the tokenized examples with the positions
    tokenized_examples.update({'start_positions': start_positions, 'end_positions': end_positions})
    return tokenized_examples

# Tokenize all datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_validation = validation_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Load your pre-trained BertForMaskedLM
if BENCH_LSH:
    masked_lm_model = BertLSHForQuestionAnswering.from_pretrained('./bertlsh')
else:
    masked_lm_model = BertForQuestionAnswering.from_pretrained('./origbert')

# Define a variable for the output directory
if BENCH_LSH:
    print("Using LSH Bert")
    output_dir = './bertlsh-squad'
else:
    print("Using Original Bert")
    output_dir = './origbert-squad'

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=os.path.join(output_dir, 'logs'),  # Directory for storing logs
    logging_strategy="steps",  # Log training loss every logging_steps
    logging_steps=50,  # Log training loss every 50 steps
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    save_strategy="steps",  # Save model checkpoints during training
    save_steps=500,  # Save checkpoint every 500 steps
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,  # Whether to load the best model (in terms of loss) at the end of training
)

# Initialize the Trainer
trainer = Trainer(
    model=masked_lm_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
)

# Train the model
trainer.train()

# After training is done, evaluate on the test set
test_results = trainer.evaluate(tokenized_test)

# Save the test results to the output directory
test_results_path = os.path.join(output_dir, 'test_results.json')
with open(test_results_path, 'w') as result_file:
    json.dump(test_results, result_file)