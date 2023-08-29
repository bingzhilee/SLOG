import csv
from typing import Dict, Optional
import logging
import copy
import re

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, Token, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("okapi")
class Seq2SeqDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    `ComposedSeq2Seq` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of `read` is a list of `Instance` s with the fields:
        source_tokens : `TextField` and
        target_tokens : `TextField`

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    # Parameters

    source_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to `SpacyTokenizer()`.
    target_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to `source_tokenizer`.
    source_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input (source side) token representations. Defaults to
        `{"tokens": SingleIdTokenIndexer()}`.
    target_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define output (target side) token representations. Defaults to
        `source_token_indexers`.
    source_add_start_token : `bool`, (optional, default=`True`)
        Whether or not to add `start_symbol` to the beginning of the source sequence.
    source_add_end_token : `bool`, (optional, default=`True`)
        Whether or not to add `end_symbol` to the end of the source sequence.
    target_add_start_token : `bool`, (optional, default=`True`)
        Whether or not to add `start_symbol` to the beginning of the target sequence.
    target_add_end_token : `bool`, (optional, default=`True`)
        Whether or not to add `end_symbol` to the end of the target sequence.
    start_symbol : `str`, (optional, default=`START_SYMBOL`)
        The special token to add to the end of the source sequence or the target sequence if
        `source_add_start_token` or `target_add_start_token` respectively.
    end_symbol : `str`, (optional, default=`END_SYMBOL`)
        The special token to add to the end of the source sequence or the target sequence if
        `source_add_end_token` or `target_add_end_token` respectively.
    delimiter : `str`, (optional, default=`"\t"`)
        Set delimiter for tsv/csv file.
    quoting : `int`, (optional, default=`csv.QUOTE_MINIMAL`)
        Quoting to use for csv reader.
    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = False,
        source_add_end_token: bool = False,
        target_add_start_token: bool = False,
        target_add_end_token: bool = False,
        start_symbol: str = START_SYMBOL,
        end_symbol: str = END_SYMBOL,
        delimiter: str = "\t",
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        quoting: int = csv.QUOTE_MINIMAL,
        add_prefix: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._source_tokenizer = source_tokenizer or WhitespaceTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers

        self._source_add_start_token = source_add_start_token
        self._source_add_end_token = source_add_end_token
        self._target_add_start_token = target_add_start_token
        self._target_add_end_token = target_add_end_token
        self._start_token: Optional[Token] = None
        self._end_token: Optional[Token] = None
        if (
            source_add_start_token
            or source_add_end_token
            or target_add_start_token
            or target_add_end_token
        ):
            if source_add_start_token or source_add_end_token:
                self._check_start_end_tokens(start_symbol, end_symbol, self._source_tokenizer)
            if (
                target_add_start_token or target_add_end_token
            ) and self._target_tokenizer != self._source_tokenizer:
                self._check_start_end_tokens(start_symbol, end_symbol, self._target_tokenizer)
        self._start_token = Token(start_symbol)
        self._end_token = Token(end_symbol)

        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.quoting = quoting

        self.add_prefix = add_prefix

    def _read(self, file_path: str):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in self.shard_iterable(
                enumerate(csv.reader(data_file, delimiter=self._delimiter, quoting=self.quoting))
            ):
                if len(row) != 2:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (row, line_num + 1)
                    )
                source_sequence, target_sequence = row
                if len(source_sequence) == 0 or len(target_sequence) == 0:
                    logger.info('skip {}th line'.format(line_num))
                    continue
                if self.add_prefix:
                    input = "parse Okapi: " + source_sequence
                else:
                    input = source_sequence
                yield self.text_to_instance(input, target_sequence)

        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )

    def text_to_instance(
        self, source_string: str,
            target_string: str = None,
    ) -> Instance:  # type: ignore
        fields: Dict[str: Field] = {}
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[: self._source_max_tokens]
        if self._source_add_start_token:
            tokenized_source.insert(0, copy.deepcopy(self._start_token))
        if self._source_add_end_token:
            tokenized_source.append(copy.deepcopy(self._end_token))
        source_field = TextField(tokenized_source)
        fields["source_tokens"] = source_field
        metadata_dict = {
            "source_text": source_string,
        }
        if target_string is not None:
            # target_string = self.replace_special_symbols(target_string)
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[: self._target_max_tokens]
            if self._target_add_start_token:
                tokenized_target.insert(0, copy.deepcopy(self._start_token))
            if self._target_add_end_token:
                tokenized_target.append(copy.deepcopy(self._end_token))
            target_field = TextField(tokenized_target)

            fields["target_tokens"] = target_field
            metadata_dict["target_text"] = target_string
            metadata_dict["target_tokens"] = [token.text for token in tokenized_target[1:-1]]

        metadata_field = MetadataField(metadata_dict)
        fields["metadata"] = metadata_field
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._target_token_indexers  # type: ignore

    def _check_start_end_tokens(
        self, start_symbol: str, end_symbol: str, tokenizer: Tokenizer
    ) -> None:
        """Check that `tokenizer` correctly appends `start_symbol` and `end_symbol` to the
        sequence without splitting them. Raises a `ValueError` if this is not the case.
        """

        tokens = tokenizer.tokenize(start_symbol + " " + end_symbol)
        err_msg = (
            f"Bad start or end symbol ('{start_symbol}', '{end_symbol}') "
            f"for tokenizer {self._source_tokenizer}"
        )
        try:
            start_token, end_token = tokens[0], tokens[-1]
        except IndexError:
            raise ValueError(err_msg)
        if start_token.text != start_symbol or end_token.text != end_symbol:
            raise ValueError(err_msg)

if __name__ == "__main__":
    plm_name = "t5-base"
    datareader = Seq2SeqDatasetReader(
                                   source_tokenizer=PretrainedTransformerTokenizer(model_name=plm_name),
                                   target_tokenizer=PretrainedTransformerTokenizer(model_name=plm_name),
                                   #  source_tokenizer=WhitespaceTokenizer(),
                                   # target_tokenizer=WhitespaceTokenizer(),
                                   source_token_indexers={
                                       "token": PretrainedTransformerIndexer(model_name=plm_name),
                                   },
                                   target_token_indexers={
                                       "token": PretrainedTransformerIndexer(model_name=plm_name),
                                       # "token": SingleIdTokenIndexer(),
                                   },
                                   add_prefix=False,
                                      # target_add_end_token=True,
                                      # target_add_start_token=True
                                   )
    max_decode_len=0
    max_source_len=0
    instances = list(datareader.read("data/okapi/orig/document/test.tsv"))
    # for ins in instances[:10]:
    #     print(ins)

    for i, instance in enumerate(instances):
        if len(instance['target_tokens']) > max_decode_len:
            max_decode_len = len(instance['target_tokens'])
        if len(instance['source_tokens']) > max_source_len:
            max_source_len = len(instance['source_tokens'])
    print(max_decode_len)
    print(max_source_len)
    print(instances[-1])
    # vocabulary = []
    # with open("data/cfq/mcd1/train.tsv") as f1:
    #     with open("data/cfq/mcd1/dev.tsv") as f2:
    #         with open("data/cfq/mcd1/test.tsv") as f3:
    #             lines = f1.readlines()+f2.readlines()+f3.readlines()
    #             for line in lines:
    #                 src, tgt = line.rstrip().split("\t")
    #                 # tgt = tgt.replace(".", " . ")
    #                 # tgt = tgt.replace(":", " : ")
    #                 # tgt = tgt.replace("/", " / ")
    #                 # tgt = tgt.replace("|", " | ")
    #                 # tgt = tgt.replace("?", " ? ")
    #                 # tgt = tgt.replace("!", " ! ")
    #                 tgt_tokens = tgt.split()
    #                 # tgt_tokens = re.findall(r"\w+|[^\w\s]", tgt, re.UNICODE)
    #                 src_tokens = src.split()
    #                 for tok in tgt_tokens+src_tokens:
    #                     if tok not in vocabulary:
    #                         vocabulary.append(tok)
    # print(vocabulary)
    # print(len(vocabulary))
    pass