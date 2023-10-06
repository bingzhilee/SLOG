# coding=utf-8
"""Convert Alto variable-free forms back into COGS logical forms."""
import re
import regex
from collections import defaultdict
import json
import pandas as pd
import sys
import string



with open('lexicon/verbs2lemmas.json') as lemma_file:
 verbs_lemmas = json.load(lemma_file)
with open('lexicon/proper_nouns.json') as propN_file:
 proper_nouns = json.load(propN_file)

def parse_varfreeLF(sent,varfreeLF):
    """parse variable free LF to extract parenthesized head/arguments pairs
    Args:
     lf: varfree lf
    Returns:
        ('head', 'label1 = argument1, label2 = argument2')
    """
    lf = get_idx_varfreeLF(sent, varfreeLF)
    stack = []
    heads = []
    previous_level_idx = 0
    for i, c in enumerate(lf):
        if c == '(':
            stack.append(i)
            level_i_lf = lf[previous_level_idx: i + 1]
            # match the string before "("
            head = re.findall(r'\w+(?= \()', level_i_lf)
            previous_level_idx = i + 1
            heads.extend(head)

        elif c == ')' and stack:
            start = stack.pop()
            arg = lf[start + 1: i].strip()
            # PCRE pattern regex that splits the string based on commas outside all parenthesis
            arg_substring_list = [x for x in regex.split(r"(\((?:[^()]++|(?1))*\))(*SKIP)(*F)|,", arg) if x]
            # match any characters as few as possible that are followed by an opening parenthesis
            pattern = '.+?(?=\()'
            match = [re.search(pattern, x).group(0).strip() if re.search(pattern, x) else x.strip() for x in
                     arg_substring_list]
            arguments = ",".join(match)
            target_head = heads.pop()
            yield (target_head, arguments)

def get_variable_name(sent_tokens):
    """Get the variable name "x _ i" for each token in the sentence."""
    variable_list = []
    for i, token in enumerate(sent_tokens):
        if token in proper_nouns:
            variable_list.append(token)
        elif token in ["Who", "What"]:
            variable_list.append("?")
        else:
            variable_list.append("x _ " + str(i))
    return variable_list

def replace_tokens_with_indexes(tokens,lf):
    """Iteratively replace tokens in the variable-free logical form with their respective indexes in the sentence
    to address issues with identical verbs.  Datasets generated using `alto-2.3.9-SNAPSHOT-all.jar` avoid this issue
    by filtering out sentences with duplicate lemmas during the generation process.
    """
    words_in_lf = lf.split()
    res_lf = ""
    token_index = 0
    for word in words_in_lf:
        if word in tokens[token_index:] and word not in ("in","on","beside"):
            idx = tokens.index(word, token_index)
            res_lf += str(idx) + " "
            token_index = idx + 1
        else:
            res_lf += word + " "
    return res_lf.strip()


def get_idx_varfreeLF(sent, lf):
    """Convert the variable free logical form into a logical form with indexes."""
    sent_tokens = sent.split()
    lemma_tokens = [verbs_lemmas[token] if token in verbs_lemmas.keys() else token for token in sent_tokens]
    idx_varfreeLF = replace_tokens_with_indexes(lemma_tokens,lf)
    c = 0
    while bool(re.search(r'= (\* )?[a-z]', idx_varfreeLF)):
        idx_varfreeLF = replace_tokens_with_indexes(lemma_tokens,idx_varfreeLF)
        c += 1
        if c > 30:
            raise Exception("too much iteration in getting idx_varfreeLF:\n {} \n compared to original:\n {}".format(idx_varfreeLF,lf))
    varfreeLF = [lemma_tokens[int(token)] if token.isnumeric() else token for token in idx_varfreeLF.split()]
    if " ".join(varfreeLF)!=lf:
        raise Exception("Error in parsing varfreeLF:\n {} \n compared to original:\n {}".format(lf," ".join(varfreeLF)))
    return idx_varfreeLF


def varfree_to_cogs_lf(sent,varfreeLF):
    """Converts the given variable free logical form into COGS logical form.
    - Nouns (entities and unaries):
        Jack --> Jack
        cat --> cat ( x _ i )
        * cat --> * cat ( x _ i )
    -  proper nouns
       eat ( agent = Jack ) --> eat . agent ( x _ 2 , Jack )
    - The variables representing common nouns:
        eat ( agent = cat ) --> cat ( x _ 1 ) AND eat . agent ( x _ 2 , x _ 1 )
    Args:
      "sent": sentence string
      "lf": variable free logical form string}
    Returns:
      The converted original cogs logical form.
    """
    sent = sent.rstrip(string.punctuation)
    tokens_list = sent.split()
    lemma_tokens = [verbs_lemmas[token] if token in verbs_lemmas.keys() else token for token in tokens_list]
    # primitives
    if len(tokens_list) == 1:
        raise Exception("The converter don't support primitive logical forms, please use 'generate_primitives.py' script")

    variable_list = get_variable_name(tokens_list)
    # maps head nodes to a list of (arg label, target node).
    head_arguments = set(parse_varfreeLF(sent,varfreeLF))
    head2args = defaultdict(list)
    for head, args_str in head_arguments:
        for child in [e.strip().split('=') for e in args_str.split(',')]:
            head2args[head].append(child)

    definite_noun_list = [False] + [True if token.lower() == "the" else False for token in tokens_list]
    indefinite_noun_list = [False] + [True if token.lower() == "a" else False for token in tokens_list]
    defini_nouns = []
    main_lf = []
    for i, token in enumerate(lemma_tokens):
        if token.lower() in ("the", "a", "that","in","on","beside"):
            continue
        # nouns part
        isDefini = definite_noun_list[i]
        isIndefini = indefinite_noun_list[i]
        if isDefini:
            defini_nouns.append("* " + token + " ( "+ variable_list[i]+" )")
        elif isIndefini:
            main_lf.append(token + " ( "+ variable_list[i]+" )")

        # remaining part
        if str(i) in head2args.keys():
            for child in head2args[str(i)]:
                if child[1].strip("* ").isnumeric():
                    sub_lf = token + " . " + child[0] + "( " + variable_list[i] + " , " + variable_list[int(child[1].strip("* "))] + " )"
                else:
                    sub_lf = token + " . " + child[0] + "( " + variable_list[i] + " ," + child[1] + " )"
                main_lf.append(sub_lf)

    if defini_nouns:
        return " ; ".join(defini_nouns) + " ; " + " AND ".join(main_lf)
    else:
        return " AND ".join(main_lf)



if __name__ == "__main__":

  """Conversion precision test using original cogs generalization set and its variable-free format from Qiu et al. 2022 """
  varfree_lf_file = "cogs_two_formats/gen_varfree_lf.tsv"
  cogs_file = "cogs_two_formats/gen_cogs_lf.tsv"
  df_varfree = pd.read_csv(varfree_lf_file, sep="\t", names=["sent", "varfree_lf","type"])
  df_cogs = pd.read_csv(cogs_file, sep="\t", names=["sent", "cogs_lf","type"])
  df_varfree["converted_lf"] = df_varfree.apply(lambda x: varfree_to_cogs_lf(x.sent, x.varfree_lf), axis=1)
  df_varfree["cogs_lf"] = df_cogs["cogs_lf"]
  exact_match = (df_varfree["converted_lf"] == df_varfree["cogs_lf"]).sum()
  total_items = df_varfree.shape[0]
  ratio = exact_match / total_items
  print(f"Exact match rate between converted LFs and original cogs LFs: {exact_match}/{total_items} ({ratio*100:.1f}%)")
