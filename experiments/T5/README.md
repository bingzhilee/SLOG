This directory contains code to run experiments with T5.

### Environment
We use Python 3.8.16. To install the required packages, run
```
pip install -r requirements.txt
```

### Running experiments
To run an experiment, run
```
./train_and_eval.sh <config> <data> <seed>
```
where `<config>` is the path to a configuration file, `<data>` is the path to a tsv data file, and `<seed>` is the random seed. For example, to run the experiment with the default configuration, run
```
./train_and_eval.sh configs/cogs_LF/T5.jsonnet ../../data/cogs_LF/gen.tsv 0
```