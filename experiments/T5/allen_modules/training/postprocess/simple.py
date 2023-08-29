
from typing import Union, Dict, Any, Optional, List
import re

from allen_modules.training.postprocess.postprocessor import Postprocessor

@Postprocessor.register("simple")
class SimplePostprocessor(Postprocessor):

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
            predicted_text = " ".join(predicted_text.split())
            output.append(predicted_text)
        return output


@Postprocessor.register("cogs")
class COGSPostprocessor(SimplePostprocessor):

    def __init__(self, segment_symbols: List[str] = None, skip_special_tokens=True,sort=False):
        """
        :param segment_symbols: symbol strings that will be segmented

        """
        super(COGSPostprocessor, self).__init__(segment_symbols, skip_special_tokens)
        self.seg_symbols = [".", ",", "="]
        self.sort = sort

    def __call__(self, predicted_texts: List[str]):
        """

        :param predicted_texts: text string list returned by allennlp models
        """
        predicted_texts = self.format(predicted_texts, segment_symbols=self.seg_symbols)

        return [self.postprocess_mr(predicted_text) for predicted_text in predicted_texts]

    def close_brackets(self, predicted_texts: str):
        tokens = predicted_texts.split()
        lb = 0
        for token in tokens:
            if token == "(":
                lb += 1
            if token == ")":
                lb -= 1
        if lb >= 1:
            for i in range(lb):
                tokens.append(")")
        return " ".join(tokens)

    def postprocess_mr(self, predicted_texts: str):
        if self.sort:
            conjuncts = re.split(r' AND | ; ', predicted_texts)
            new_conjuncts = sorted(conjuncts)
            predicted_texts = " AND ".join(new_conjuncts)

        predicted_texts = self.close_brackets(predicted_texts)

        return predicted_texts

@Postprocessor.register("cfq")
class CFQPostprocessor(SimplePostprocessor):

    def __init__(self, segment_symbols: List[str] = None, skip_special_tokens=True, sort=False, sort_use_brackets=False):
        """
        :param segment_symbols: symbol strings that will be segmented

        """
        super(CFQPostprocessor, self).__init__(segment_symbols, skip_special_tokens)
        self.seg_symbols = [",", "(", ")"]
        self.sort = sort
        self.use_brackets = sort_use_brackets

    def __call__(self, predicted_texts: List[str]):

        predicted_texts = self.format(predicted_texts, segment_symbols=self.seg_symbols)

        return [self.postprocess_program(predicted_text) for predicted_text in predicted_texts]

    def _get_program_parts(self, program):
        """
        This functions is copied from https://github.com/google-research/language/blob/e2b407d154bdbfa4a0718e68cefa4aa1a28c2471/language/compir/dataset_parsers/cfq_parser.py#L43
        Parses a SPARQL program into a prefix and conjuncts.
        """
        # Remove the closing bracket and split on opening bracket.
        if not program.endswith(" rb"):
            raise ValueError("Wrong program format.")
        program_no_closing = program[:-3]
        parts = program_no_closing.split(" lb ")
        if len(parts) != 2:
            raise ValueError("Wrong program format.")
        prefix = parts[0]
        conjuncts_str = parts[1]
        conjuncts = conjuncts_str.split(" . ")
        return prefix, conjuncts

    def remove_brackets_from_conjuncts(self, conjunct):
        conj = conjunct.replace("(", "").replace(")", "")
        return " ".join(conj.split())


    def postprocess_program(self, program):
        """
        This functions is copied from https://github.com/google-research/language/blob/e2b407d154bdbfa4a0718e68cefa4aa1a28c2471/language/compir/dataset_parsers/cfq_parser.py
        Postprocesses a predicted SPARQL program.
        """
        if not program.endswith(" rb") or len(program.split(" lb ")) != 2:
            return program
        prefix, conjuncts = self._get_program_parts(program)
        # Take unique conjuncts and sort them alphabetically. FILTER conjuncts can
        # have duplicates, so these are not turned into a set.
        conjuncts_unique = [conjunct for conjunct in conjuncts if not conjunct.startswith("FILTER")] \
                           + [conjunct for conjunct in conjuncts if conjunct.startswith("FILTER")]
        # print("#######CONJUNCT########")
        # print(conjuncts_unique)
        if self.sort and not self.use_brackets:
            conjuncts_ordered = sorted(list(conjuncts_unique), key=lambda x: self.remove_brackets_from_conjuncts(x))
        elif self.sort and self.use_brackets:
            conjuncts_ordered = sorted(list(conjuncts_unique), key=lambda x: x)
        else:
            conjuncts_ordered = list(conjuncts_unique)

        # print(conjuncts_ordered)
        # raise NotImplementedError

        program_processed = "{} lb {} rb".format(prefix,
                                                 " . ".join(conjuncts_ordered))
        # # Replace back T5 OOV tokens.
        # program_processed = program_processed.replace("lb", "{")
        # program_processed = program_processed.replace("rb", "}")
        # program_processed = program_processed.replace("#", "^")
        return program_processed