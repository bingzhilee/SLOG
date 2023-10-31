# SLOG
This repository contains data and code for [SLOG](https://arxiv.org/abs/2310.15040), a **S**tructural **LO**ng-distance dependencies **G**eneralization benchmark. SLOG is a semantic parsing dataset that extends the COGS benchmark ([Kim and Linzen,2020](https://aclanthology.org/2020.emnlp-main.731/)) with 17 *structural generalization* tasks. 

## Dataset
- The training sets and generalization sets can be found under [`/data`](data), please note that access to the generalization sets is restricted with a password-protected Zip file, with the password being: SLOG. This precaution is to prevent the sets from being unintentionally included in the training data of LLMs, ensuring they remain valid tools for unbiased model evaluation.

- The third column of the generalization set files specifies the 17 generalization types. For a comprehensive description with illustrative examples, please refer to Table 2 in the paper.   

- The code for generating SLOG is under [`/generation_scripts`](generation_scripts) directory.

## Experiments
On SLOG, we trained a vanilla Transformer from scratch [`/experiments/vanilla_transformer`](experiments/vanilla_transformer), finetuned a pretrained T5-base: [`/experiments/T5`](experiments/T5) and a pretrained LLaMa [`/experiments/llama`](experiments/llama). For hyperparameters and random seed details of each model, please see the respective directories.

