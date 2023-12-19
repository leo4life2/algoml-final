---
base_model: leo4life/algoml-bert-tiny
tags:
- generated_from_trainer
datasets:
- leo4life/algoml_bookcorpus_10p
metrics:
- accuracy
model-index:
- name: bertlsh
  results:
  - task:
      name: Masked Language Modeling
      type: fill-mask
    dataset:
      name: leo4life/algoml_bookcorpus_10p
      type: leo4life/algoml_bookcorpus_10p
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.18187143638209793
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bertlsh

This model is a fine-tuned version of [leo4life/algoml-bert-tiny](https://huggingface.co/leo4life/algoml-bert-tiny) on the leo4life/algoml_bookcorpus_10p dataset.
It achieves the following results on the evaluation set:
- Loss: 5.3308
- Accuracy: 0.1819

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.37.0.dev0
- Pytorch 2.1.2+cu121
- Datasets 2.15.0
- Tokenizers 0.15.0
