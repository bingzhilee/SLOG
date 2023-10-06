We used the codebase from [Csordas et al. 2021](https://github.com/robertcsordas/transformer_generalization) for experiments with Transformer training from scratch (vanilla transformer). The model architecture is the same as in original COGS, which consists of 2 encoder and 2 decoder layers, 4 attention heads per layer, and a feedforward dimension of 512.

### Setup
Clone the repository and install the necessary requirements. This creates a folder named `transformer_generalization` under your current working directory. Change to that folder. 
```
git clone https://github.com/RobertCsordas/transformer_generalization.git
```

To train and evaluate a Transformer on SLOG benchmark:
-  Update the `URL_BASE` class attribute in the `transformer_generalization/dataset/text/cogs.py`file to the path of the SLOG dataset. 
```
URL_BASE = "https://raw.githubusercontent.com/bingzhilee/SLOG/main/data/cogs_LF/"
```
- Copy the `./slog_trafo.yaml` configuration file, which contains the hyperparameters used in our paper, to the `transformer_generalization/sweeps/` directory.
