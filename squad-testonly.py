from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, EvalPrediction, BertForQuestionAnswering
from transformers.models.bert.modeling_lsh_bert import BertLSHForQuestionAnswering
import os
import json
from evaluate import load

BENCH_LSH = False

# Load the SQuAD dataset
full_dataset = load_dataset('squad')
test_dataset = full_dataset['validation']

# Load the tokenizer
if BENCH_LSH:
    tokenizer = AutoTokenizer.from_pretrained('./bertlsh')
    model = BertLSHForQuestionAnswering.from_pretrained('./bertlsh-squad')
else:
    tokenizer = AutoTokenizer.from_pretrained('./origbert')
    model = BertForQuestionAnswering.from_pretrained('./origbert-squad')

# Tokenize the test dataset
def tokenize_function(examples):
    return tokenizer(
        examples['question'], examples['context'],
        padding='max_length', truncation=True, max_length=384,
        return_tensors='pt'
    )

tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Load SQuAD metric
squad_metric = load("squad")

def compute_metrics(p: EvalPrediction):
    # Convert logits to start and end positions
    start_logits, end_logits = p.predictions
    start_positions = start_logits.argmax(-1)
    end_positions = end_logits.argmax(-1)

    predictions = []
    references = []

    for i in range(len(p.label_ids)):
        # Convert token positions back to text
        start = start_positions[i]
        end = end_positions[i]
        prediction_text = tokenizer.decode(tokenized_test[i]['input_ids'][start:end+1], skip_special_tokens=True)
        predictions.append({'prediction_text': prediction_text, 'id': test_dataset[i]['id']})

        # References
        references.append({'answers': test_dataset[i]['answers'], 'id': test_dataset[i]['id']})

    return squad_metric.compute(predictions=predictions, references=references)

# Define the output directory
output_dir = './bertlsh-squad' if BENCH_LSH else './origbert-squad'

# Training arguments for evaluation
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=64,
    do_train=False,
    do_eval=True
)

# Initialize Trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

# Evaluate the model on the test set
test_results = trainer.evaluate()

# Save the test results to the output directory
test_results_path = os.path.join(output_dir, 'test_results.json')
with open(test_results_path, 'w') as result_file:
    json.dump(test_results, result_file)

print("Evaluation on test set complete. Results saved to:", test_results_path)
