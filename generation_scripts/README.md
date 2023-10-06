## SLOG generation
We used a probabilistic Synchronous Context-Free Grammar (SCFG) implemented in [Alto](https://github.com/coli-saar/alto) to simultaneously generate English sentences and their corresponding meaning representations. For detailed description of the grammar implementation, please refer to [this page](https://github.com/bingzhilee/SLOG/wiki/Reimplementation-of-the-COGS-grammar-for-Alto).

Below are the steps to generate the SLOG corpus:
1. We use python 3.9.17. To install the required packages, run:
```
pip install -r requirements.txt
```

2. Run`cogs-preprocess.py` in the `grammars` directory to convert specific [Jinja](https://palletsprojects.com/p/jinja/) templates into corresponding [IRTG](https://github.com/coli-saar/alto/wiki/GettingStarted) grammars:

```
python cogs-preprocess.py specify_grammar.irtg > preprocessed-<grammar-name>.irtg
```

3. Load `preprocessed-main.irtg` into Alto to generate pairs of sentences and [variable-free format LFs](https://github.com/google-research/language/tree/master/language/compgen/csl)(Qiu et al. 2022): 
```bash
java -cp ../alto-2.3.9-SNAPSHOT-all.jar de.up.ling.irtg.script.CogsCorpusGenerator \
         --count 1000 \
         --suppress-duplicates \
         --pp-depth 0-2 \
         --cp-depth 0-2 \
         --cemb-depth 0-2 preprocessed_PP_modif_iobj_gen.irtg > alto_PP_modif_iobj_gen.tsv
```
> where `../alto-2.3.9-SNAPSHOT-all.jar` bundles Alto classes and all dependent libraries. For detailed documentation, see [this page](https://github.com/bingzhilee/SLOG/wiki/Alto-source-code). The output tsv file has two columns: English sentence and variable-free meaning representation. The options are as follows:
>- `--count <N>` says that we want to generate a corpus with `<N>` instances.
>- `--suppress-duplicates` says that the same sentence should never be generated twice. 
>- `--pp-depth <min>-<max>` restricts the PP embedding depth to a minimum of `<min>` and a maximum of `<max>`. For instance, write `--pp-depth 0-2` to generate instances with PP depth at most two.
>- `--cp-depth <min>-<max>` restricts the CP embedding depth in the same way. 
>- `--cemb-depth <min>-<max>` restricts the center-embedding depth in the same way.

> For further details on additional options, please refer to the [Alto documentation](https://github.com/coli-saar/alto/wiki/Generating-a-COGS-corpus). 

4. Postprocess alto output and convert the variable-free format to variable-based format (cogs format):
```
python varfree2cogs_converter/alto_output_to_two_lfs.py alto_PP_modif_iobj_gen.tsv PP_modif_iobj
```
