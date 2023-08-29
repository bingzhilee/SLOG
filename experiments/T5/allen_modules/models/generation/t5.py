from os import PathLike
from typing import Optional, Dict, Any, Union, List, Tuple

import numpy as np
import torch
# Set the maximal number of CPU cores
torch.set_num_threads(4)

from allennlp.common.lazy import Lazy
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models.model import Model
# from allennlp.modules.transformer.t5 import T5 as T5Module
from allennlp.modules.transformer.t5 import T5Output, IntT, BoolT
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.checkpoint import CheckpointWrapper
from allennlp.training.metrics import ROUGE, BLEU

from allen_modules.training.metrics.exact_match import ExactMatchAcc
from allen_modules.training.metrics.epoch import EpochsPassed
from allen_modules.training.postprocess.postprocessor import Postprocessor
from allen_modules.training.postprocess.simple import SimplePostprocessor
from allen_modules.modules.transformer.t5 import T5 as T5Module

@Model.register("modified_t5")
class T5(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch, beam_size=3, max_steps=50),
        checkpoint_wrapper: Optional[CheckpointWrapper] = None,
        weights_path: Optional[Union[str, PathLike]] = None,
        postprocessor: Postprocessor = None,
        print_err: bool = False,
        val_epoch: bool = False,
        val_bleu: bool = False,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._model_name = model_name
        # We only instantiate this when we need it.
        self._tokenizer: Optional[PretrainedTransformerTokenizer] = None

        self.t5 = T5Module.from_pretrained_module(
            model_name,
            beam_search=beam_search,
            ddp_accelerator=self.ddp_accelerator,
            checkpoint_wrapper=checkpoint_wrapper,
            weights_path=weights_path,
        )
        # print("Set beam size to 4")
        # print(self.t5.beam_search.beam_size)
        # self.t5.beam_search.beam_size = 4
        # assert self.t5.beam_search.beam_size == 4
        # if freeze_decoder:
        #     for parameter in self.t5.decoder.parameters():
        #         parameter.requires_grad = False

        exclude_indices = {
            self.t5.pad_token_id,
            self.t5.decoder_start_token_id,
            self.t5.eos_token_id,
        }
        self.postprocessor = postprocessor
        # Use exact match accuracy as main validation metric
        self._acc = ExactMatchAcc(print_err=print_err)
        self._metrics = [self._acc]

        # For most experiments, we want to maintain epoch number to train for a certain number of epochs
        self.val_epoch = val_epoch
        if self.val_epoch:
            self._epochs = EpochsPassed()
            self._metrics.append(self._epochs)

        self.val_bleu = val_bleu
        if self.val_bleu:
            self._bleu = BLEU(exclude_indices=exclude_indices)
            self._metrics.append(self._bleu)


    def _post_load_state_dict(
        self, missing_keys: List[str], unexpected_keys: List[str]
    ) -> Tuple[List[str], List[str]]:
        missing_keys_to_ignore = [
            "t5.encoder.token_embeddings.weight",
            "t5.decoder.token_embeddings.weight",
        ]
        if self.t5._tie_word_embeddings:
            missing_keys_to_ignore.append("t5.lm_head.weight")
        for key in missing_keys_to_ignore:
            if key in missing_keys:
                missing_keys.remove(key)
        return missing_keys, unexpected_keys

    @property
    def tokenizer(self) -> PretrainedTransformerTokenizer:
        if self._tokenizer is None:
            self._tokenizer = PretrainedTransformerTokenizer(self._model_name)
        return self._tokenizer

    def forward(  # type: ignore
        self, source_tokens: TextFieldTensors,
            target_tokens: Optional[TextFieldTensors] = None,
            metadata: Dict = None
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward step of T5.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The source tokens for the encoder. We assume they are stored under the `tokens` key/namespace.

        target_tokens : `TextFieldTensors`, optional (default = `None`)
            The target tokens for the decoder. We assume they are also stored under the `tokens` key/namespace.
            If no target tokens are given during training / validation, the source tokens are shifted
            to the right by 1.

        # Returns

        `Dict[str, torch.Tensor]`
            Contains the `loss` when `target_tokens` is provided.
            And during prediction, includes `predictions` and `predicted_log_probs` from beam search.

        """
        input_ids, attention_mask = (
            source_tokens["tokens"]["token_ids"],
            source_tokens["tokens"]["mask"],
        )
        labels: Optional[IntT] = None
        decoder_attention_mask: Optional[BoolT] = None
        if target_tokens is not None:
            labels, decoder_attention_mask = (
                target_tokens["tokens"]["token_ids"],  # type: ignore[assignment]
                target_tokens["tokens"]["mask"],  # type: ignore[assignment]
            )
        elif self.training:
            raise ValueError("'target_tokens' required during training")

        output: T5Output = self.t5(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        output_dict: Dict[str, torch.Tensor] = {}

        if self.training:
            assert output.loss is not None
            output_dict["loss"] = output.loss
        else:
            # Shape: (batch_size, beam_size, num_tokens)
            assert output.predictions is not None
            # Shape: (batch_size, beam_size)
            assert output.predicted_log_probs is not None
            # Shape: (batch_size, num_tokens)
            output_dict["predictions"] = output.predictions[:, 0, :]
            # Shape: (batch_size, )
            output_dict["predicted_log_probs"] = output.predicted_log_probs[:, 0]

            output_dict = self.make_output_human_readable(output_dict)

            if self.val_epoch:
                self._epochs()

            if labels is not None:
                # Compute exact match accuracy as main validation metric
                self._acc(output_dict["predicted_text"], metadata)

                if self.val_bleu:
                    self._bleu(output_dict["predictions"], labels)

                # Save loss of each instance for ece computation and output
                assert output.loss is not None
                output_dict["loss"] = output.loss.mean()
                batch_size, seq_len = labels.size()
                loss_per_ins = output.loss.view(batch_size, seq_len)
                loss_per_ins = torch.sum(loss_per_ins, dim=1, keepdim=False)
                output_dict["loss_per_ins"] = loss_per_ins.tolist()
                output_dict["gold_probs"] = torch.exp(-1 * loss_per_ins).tolist()


        return output_dict

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # print(output_dict.keys())
        predictions = output_dict["predictions"]

        predicted_texts = self.tokenizer.tokenizer.batch_decode(
            predictions, skip_special_tokens=self.postprocessor.skip_special_tokens if self.postprocessor is not None else True, clean_up_tokenization_spaces=False  # type: ignore[attr-defined]
        )
        if self.postprocessor is not None:
            output_dict["predicted_text"] = self.postprocessor(predicted_texts)
        else:
            output_dict["predicted_text"] = predicted_texts

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not self.training:
            for metric in self._metrics:
                metrics.update(metric.get_metric(reset=reset))
        return metrics

    @classmethod
    def from_archive(cls, archive_file: str, vocab: Vocabulary = None, weights_file:str = None,
                     beam_search:Dict=None) -> "Model":
        """
        Loads a model from an archive file.  This basically just calls
        `return archival.load_archive(archive_file).model`.  It exists as a method here for
        convenience, and so that we can register it for easy use for fine tuning an existing model
        from a config file.

        If `vocab` is given, we will extend the loaded model's vocabulary using the passed vocab
        object (including calling `extend_embedder_vocab`, which extends embedding layers).
        """
        from allennlp.models.archival import load_archive  # here to avoid circular imports

        model = load_archive(archive_file, weights_file=weights_file).model
        if vocab:
            model.vocab.extend_from_vocab(vocab)
            model.extend_embedder_vocab()
        if beam_search is not None:
            beam_size = beam_search["beam_size"]
            model.t5.beam_search.beam_size = beam_size
            print("set beam size to {}".format(beam_size))
        # raise NotImplementedError
        return model

    default_predictor = "seq2seq"

Model.register("from_archive_T5_beam", constructor="from_archive")(T5)
