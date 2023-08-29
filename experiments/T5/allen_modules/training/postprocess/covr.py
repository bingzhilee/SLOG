
from typing import Union, Dict, Any, Optional, List
import re
import copy
from allen_modules.training.postprocess.postprocessor import Postprocessor
from allen_modules.training.postprocess.simple import SimplePostprocessor

@Postprocessor.register("covr")
class COVRPostprocessor(SimplePostprocessor):

    def __init__(self, segment_symbols: List[str] = None, skip_special_tokens=True):
        """
        :param segment_symbols: symbol strings that will be segmented

        """

        self.seg_symbols = segment_symbols
        self.skip_special_tokens = skip_special_tokens

    def __call__(self, predicted_texts: List[str]):
        """

        :param predicted_texts: text string list returned by allennlp models
        """
        return self.format(predicted_texts, segment_symbols=self.seg_symbols)

    def format(self, predicted_texts: List[str],
                    segment_symbols: List[str] = None,
                    remove_symbols: List[str] = None):
        """

        :param predicted_texts: text string list returned by allennlp models
        """
        output = []
        for predicted_text in predicted_texts:
            predicted_text = predicted_text.rstrip().lstrip()
            if segment_symbols is not None:
                for sym in segment_symbols:
                    predicted_text = predicted_text.replace(sym, " {} ".format(sym))
            if remove_symbols is not None:
                for sym in remove_symbols:
                    predicted_text = predicted_text.replace(sym, "")
            predicted_text = self.close_brackets(predicted_text)
            predicted_text = self.reconstruct_rc_consts(predicted_text)
            predicted_text = " ".join(predicted_text.split())
            output.append(predicted_text)
        return output

    def close_brackets(self, predicted_text):
        tokens = predicted_text.split()
        bracketed = 0
        for i in range(len(tokens)):
            if tokens[i] == "(":
                bracketed += 1
            elif tokens[i] == ")":
                bracketed -= 1
        if bracketed > 0:
            tokens += [")" for _ in range(bracketed)]
        elif bracketed < 0:
            if tokens[len(tokens)+bracketed:] != [")" for _ in range(bracketed)]:
                return predicted_text
            tokens = tokens[:len(tokens)+bracketed]
        return " ".join(tokens)

    def reconstruct_rc_consts(self, predicted_text:str):
        def find_closed_brackets(tokens, idx):
            bracketed = 0
            for i in range(idx, len(tokens)):
                if tokens[i] == "(":
                    bracketed += 1
                elif tokens[i] == ")":
                    bracketed -= 1
                if bracketed == 0:
                    return i
            # print(tokens, idx)
            raise AssertionError

        def find_subconsts(tokens):
            subconst_lists = []
            i = 0
            const_start_idx = 0
            # print(len(tokens))
            print(tokens)
            while i < len(tokens):
                # print(i)
                if tokens[i] == "(":
                    end_idx = find_closed_brackets(tokens, i)
                    # print(end_idx)
                    subconst_lists.append(tokens[const_start_idx:end_idx + 1])
                    # print(end_idx)
                    if end_idx + 1 != len(tokens):
                        assert tokens[end_idx + 1] == ","
                        const_start_idx = end_idx + 2
                    else:
                        break
                    i = end_idx + 2
                elif tokens[i] == ",":
                    subconst_lists.append(tokens[const_start_idx:i])
                    const_start_idx = i + 1
                    i += 1
                else:
                    i += 1
            if tokens and tokens[-1] != ")":
                subconst_lists.append(tokens[const_start_idx:])
            # print(subconst_lists)
            # print(terminal_list)
            # raise NotImplementedError
            return subconst_lists

        class covr_tree(object):
            def __init__(self, covr_tokens):
                if "(" not in covr_tokens or covr_tokens == ["scene", "(", ")"]:
                    const_sym = " ".join(covr_tokens)
                    self.node = const_sym
                    self.childs = []
                else:
                    if covr_tokens[1] != "[":
                        const_sym = covr_tokens[0]
                        left_bracket = 1
                    else:
                        const_sym = " ".join(covr_tokens[:4])
                        left_bracket = 4

                    self.node = const_sym
                    self.childs = []

                    subconsts = find_subconsts(covr_tokens[left_bracket+1:-1])
                    for subconst in subconsts:
                        self.childs.append(covr_tree(subconst))

            def to_string(self):
                if not self.childs:
                    return self.node
                subtree_strings = " , ".join([child.to_string() for child in self.childs])
                tree_string_tokens = [self.node, "("]+subtree_strings.split()+[")"]
                return " ".join(tree_string_tokens)

            def add_subtree(self, subtree):
                assert isinstance(subtree, covr_tree)
                self.childs.append(subtree)

            def replace_subtree(self, subtree, idx=0):
                # print(subtree)
                assert isinstance(subtree, covr_tree)
                self.childs[idx] = subtree


            def restruct_rc(self):
                if not self.childs:
                    return self
                for i,subtree in enumerate(self.childs):
                    # print(self.node)
                    self.childs[i] = subtree.restruct_rc()
                if self.node == "filter" and self.childs[1].node == "with_relation":
                    # if self.childs[1].childs[0] is None:
                    #     print(self.node)
                    filter_subtree = copy.deepcopy(self)
                    filter_subtree.replace_subtree(self.childs[1].childs[0], idx=1)
                    new_tree = copy.deepcopy(self.childs[1])
                    new_tree.replace_subtree(filter_subtree, idx=0)
                else:
                    new_tree = self
                return new_tree
        tokens = predicted_text.split()
        try:
            tree = covr_tree(tokens)
        except AssertionError:
            return " ".join(tokens)
        newtree = tree.restruct_rc()
        return newtree.to_string()

if __name__ == "__main__":
    processor = COVRPostprocessor()
    mr = "count ( filter ( round , with_relation ( find ( animal ) , looking at , filter ( white , find ( mouse ) ) ) ) )"
    new_mr = processor.reconstruct_rc_consts(mr)
    print(new_mr)