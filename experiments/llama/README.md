This directory contains code to run experiments with llama.

### Environment
We use Python 3.8.16. To install the required packages, run
```
pip install -r requirements.txt
```

### Running experiments
To run an experiment, first run the following command to generate the data with json format:
```
python convert_tsv_to_json.py
```
Then, run the command below to fine-tune the model:
```
./run_scripts/finetune_cogs_LF.sh <seed>
```
where `<seed>` is the random seed.

To evaluate the model, run
```
./run_scripts/evaluate_cogs_LF.sh <path>
```
where `<path>` is the path to the directory of model checkpoint.