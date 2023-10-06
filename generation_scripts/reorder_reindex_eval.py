# coding=utf-8

"""Convert cogs lf forms into linear index lf to apply new evaluations """

import sys
import pandas as pd
import re

def token_removal(lf):
    rm_tokens = ['x', '_']
    terms = []
    for t in lf.split():
        if t not in rm_tokens:
            terms += [t]
    ret = " ".join(terms).strip()
    return ret

def convert_lf_to_ignoring_conjounct_order(lf):
    # single conjunct
    if " ; " not in lf and " AND " not in lf:
        return lf

    # split lf into conjuncts first based on " ; "
    conjuncts = lf.split(" ; ")
    if len(conjuncts) == 2 and " AND " not in conjuncts[0] and " AND " not in conjuncts[1]:
        return lf
    # exclude conjuncts with " ; " in them
    main_lf = [conj for conj in conjuncts if " AND " in conj]
    defini_nouns = [conj for conj in conjuncts if " AND " not in conj]
    # split conjuncts with " AND " into a list of conjuncts
    if len(main_lf) < 1 and not defini_nouns:
        breakpoint()
    if main_lf:
        main_lf = main_lf[0].split(" AND ")
        # Sort the list of conjuncts based on the first letter and then the second letter of each string
        main_lf.sort(key=lambda s: tuple(s[i] for i in range(len(s))))

    defini_nouns.sort(key=lambda s: tuple(s[i] for i in range(len(s))))
    if len(defini_nouns) > 0:
        return " ; ".join(defini_nouns) + " ; " + " AND ".join(main_lf)
    else:
        return " AND ".join(main_lf)


def reindex_reorder(lf):
    lf_simp = token_removal(lf)
    new_lf = []
    old_index2new = {}
    current_index = 1
    for token in lf_simp.split():
        if token.isnumeric():
            if token not in old_index2new:
                old_index2new[token] = current_index
                new_lf += [str(current_index)]
                current_index += 1
            else:
                new_lf += [str(old_index2new[token])]
        else:
            new_lf += [token]

    new_lf = " ".join(new_lf)
    lf_reorder = convert_lf_to_ignoring_conjounct_order(new_lf)
    if lf_reorder:
        return lf_reorder
    else:
        return new_lf


def reorder_reindex(lf):
    lf_simp = token_removal(lf)
    lf_reorder = convert_lf_to_ignoring_conjounct_order(lf_simp)
    new_lf = []
    old_index2new = {}
    current_index = 1
    for token in lf_reorder.split():
        if token.isnumeric():
            if token not in old_index2new:
                old_index2new[token] = current_index
                new_lf += [str(current_index)]
                current_index += 1
            else:
                new_lf += [str(old_index2new[token])]
        else:
            new_lf += [token]

    new_lf = " ".join(new_lf)

    return new_lf