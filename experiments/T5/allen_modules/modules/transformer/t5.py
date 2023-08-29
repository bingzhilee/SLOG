import logging
from typing import Optional, Tuple, List, Union, Dict, TYPE_CHECKING, NamedTuple, Callable

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from allennlp.common import FromParams, Params, Lazy, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.attention_module import (
    T5Attention,
    AttentionOutput,
)
from allennlp.modules.transformer.t5 import *
from allennlp.modules.transformer.util import (
    get_extended_attention_mask,
    FloatT,
    IntT,
    BoolT,
)
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.parallel import DdpAccelerator
from allennlp.nn.checkpoint import CheckpointWrapper

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


class T5(TransformerModule, Registrable):

    _pretrained_mapping = {"shared": "token_embeddings"}

    # Don't know why HF has this param in their state_dict. It's not used in their model.
    _pretrained_ignore = [
        r"^decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight$"
    ]

    default_implementation = "default"

    def __init__(
        self,
        token_embeddings: Optional[nn.Embedding] = None,
        encoder: Lazy[T5EncoderStack] = Lazy(T5EncoderStack.basic_encoder),
        decoder: Lazy[T5DecoderStack] = Lazy(T5DecoderStack.basic_decoder),
        decoder_start_token_id: int = 0,
        pad_token_id: int = 0,  # These are both 0 in t5-(small|base|large). Go figure.
        eos_token_id: int = 1,
        vocab_size: int = 32128,
        model_dim: int = 512,
        output_attentions: bool = False,
        output_all_hidden_states: bool = False,
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch, beam_size=3, max_steps=100),
        ddp_accelerator: Optional[DdpAccelerator] = None,
        checkpoint_wrapper: Optional[CheckpointWrapper] = None,
        tie_word_embeddings: bool = True,
        label_smoothing: float = None,
    ):
        super().__init__()
        self._tie_word_embeddings = tie_word_embeddings

        self.model_dim = model_dim
        self.token_embeddings = token_embeddings or nn.Embedding(vocab_size, model_dim)
        if token_embeddings is None:
            self.token_embeddings.weight.data.normal_(mean=0.0, std=1.0)
        self.encoder: T5EncoderStack = encoder.construct(
            token_embeddings=self.token_embeddings,
            ddp_accelerator=ddp_accelerator,
            checkpoint_wrapper=checkpoint_wrapper,
        )
        self.decoder: T5DecoderStack = decoder.construct(
            token_embeddings=self.token_embeddings,
            ddp_accelerator=ddp_accelerator,
            checkpoint_wrapper=checkpoint_wrapper,
        )
        self.lm_head = nn.Linear(
            self.decoder.hidden_size, self.token_embeddings.num_embeddings, bias=False
        )
        if self._tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self.loss_fct = CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing if label_smoothing is not None else 0.0)
        self.loss_fct_batch = CrossEntropyLoss(ignore_index=-100, reduction="none")

        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.output_attentions = output_attentions
        self.output_all_hidden_states = output_all_hidden_states

        self.beam_search = beam_search.construct(end_index=self.eos_token_id)

        
    def resize_token_embeddings(
        self, new_size: int, *, init_fn: Callable = torch.nn.init.normal_
    ) -> None:
        """
        Resizes the token embeddings in the model.

        This takes care of the token embeddings for the encoder, the decoder, and the LM head.

        new_size : `int`
            The new size of the token embeddings
        init_fn : `Callable`
            The function to use to initialize new embeddings. This function will be called with a
            single argument, the tensor to initialize, and it is expected to initialize the tensor
            in place. Many of the functions from `torch.nn.init` fit.
        """
        self.encoder.resize_token_embeddings(new_size, init_fn=init_fn)
        # If encoder and decoder share embeddings, this is a no-op the second time.
        self.decoder.resize_token_embeddings(new_size, init_fn=init_fn)

        # resize lm head
        old_size = self.lm_head.out_features
        if old_size == new_size:
            return
        new_lm_head = torch.nn.Linear(
            self.lm_head.in_features,
            new_size,
            self.lm_head.bias,
            self.lm_head.weight.device,
            self.lm_head.weight.dtype,
        )
        copy_size = min(old_size, new_size)
        new_lm_head.weight.data[:copy_size, ...] = self.lm_head.weight.data[:copy_size, ...]
        if self.lm_head.bias and new_lm_head.bias:
            new_lm_head.bias.data[:copy_size, ...] = self.lm_head.bias[:copy_size, ...]
        if new_size > old_size:
            init_fn(new_lm_head.weight.data[copy_size:, ...])
            if new_lm_head.bias:
                init_fn(new_lm_head.bias[copy_size:, ...])

        self.lm_head = new_lm_head

    def _post_load_state_dict(
        self, missing_keys: List[str], unexpected_keys: List[str]
    ) -> Tuple[List[str], List[str]]:
        missing_keys_to_ignore = [
            "encoder.token_embeddings.weight",
            "decoder.token_embeddings.weight",
        ]
        if self._tie_word_embeddings:
            missing_keys_to_ignore.append("lm_head.weight")
        for key in missing_keys_to_ignore:
            if key in missing_keys:
                missing_keys.remove(key)
        return missing_keys, unexpected_keys

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs):
        attention_kwargs = {
            "hidden_size": config.d_model,
            "key_value_proj_dim": config.d_kv,
            "num_heads": config.num_heads,
            "relative_attention_num_buckets": config.relative_attention_num_buckets,
            "dropout": config.dropout_rate,
        }
        layer_norm_kwargs = {
            "hidden_size": config.d_model,
            "eps": config.layer_norm_epsilon,
        }
        block_ff = Lazy(
            T5LayerFF,
            params=Params(
                {
                    "ff_proj": {
                        "type": config.feed_forward_proj,
                        "hidden_size": config.d_model,
                        "ff_size": config.d_ff,
                        "dropout": config.dropout_rate,
                    },
                    "layer_norm": layer_norm_kwargs,
                    "dropout": config.dropout_rate,
                }
            ),
        )
        return cls(
            encoder=Lazy(
                T5EncoderStack.basic_encoder,
                constructor_extras={
                    "num_blocks": config.num_layers,
                    "block_self_attention": Lazy(T5Attention, constructor_extras=attention_kwargs),
                    "final_layer_norm": T5LayerNorm(**layer_norm_kwargs),
                    "block_ff": block_ff,
                    "dropout": config.dropout_rate,
                },
            ),
            decoder=Lazy(
                T5DecoderStack.basic_decoder,
                constructor_extras={
                    "num_blocks": config.num_decoder_layers,
                    "block_self_attention": Lazy(T5Attention, constructor_extras=attention_kwargs),
                    "block_cross_attention": Lazy(T5Attention, constructor_extras=attention_kwargs),
                    "final_layer_norm": T5LayerNorm(**layer_norm_kwargs),
                    "block_ff": block_ff,
                    "dropout": config.dropout_rate,
                },
            ),
            decoder_start_token_id=config.decoder_start_token_id,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            vocab_size=config.vocab_size,
            model_dim=config.d_model,
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", config.tie_word_embeddings),
            **kwargs,
        )

    def _shift_right(self, input_ids, start_value: int):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = start_value

        return shifted_input_ids

    def _get_lm_logits(self, decoder_last_hidden_state: FloatT) -> FloatT:
        # Shape: (batch_size, target_length, model_dim)
        sequence_output = decoder_last_hidden_state
        # Rescale output before projecting on vocab
        # TODO: HF only does this when does this when embeddings are tied.
        # Currently tied embeddings is the only option we have, but if make
        # that configurable then we should put this in an 'if' block.
        sequence_output = sequence_output * (self.model_dim**-0.5)
        # Shape: (batch_size, target_length, vocab_size)
        logits = self.lm_head(sequence_output)
        return logits

    def forward(
        self,
        input_ids: IntT,
        attention_mask: Optional[BoolT] = None,
        labels: Optional[IntT] = None,
        decoder_attention_mask: Optional[BoolT] = None,
    ) -> T5Output:
        """
        Run forward pass of the model.
        """
        if attention_mask is None:
            attention_mask = ~(input_ids == self.pad_token_id)

        # Encode inputs.
        encoder_outputs: T5StackOutput = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            output_all_hidden_states=self.output_all_hidden_states,
        )

        logits: Optional[FloatT] = None
        loss: Optional[FloatT] = None
        decoder_outputs: Optional[T5StackOutput] = None
        predictions: Optional[IntT] = None
        predicted_log_probs: Optional[FloatT] = None

        if labels is not None:
            # Calculate loss against targets.

            if decoder_attention_mask is None:
                decoder_attention_mask = ~(labels == self.pad_token_id)

            # Get decoder inputs from shifting lm labels to the right and pre-pending
            # the decoder start token ID.
            # Shape (both): (batch_size, target_length)
            decoder_input_ids = self._shift_right(labels, self.decoder_start_token_id)

            # Replace possible -100 values in labels by `pad_token_id`
            decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id)

            # Decode.
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                output_attentions=self.output_attentions,
                output_all_hidden_states=self.output_all_hidden_states,
            )

            # Shape: (batch_size, target_length, vocab_size)
            logits = self._get_lm_logits(decoder_outputs.last_hidden_state)  # type: ignore[union-attr]

            if self.training:
                # Shape: (1,)
                # if self.label_smoothing is not None and self.label_smoothing > 0.0:
                    # flat_logits = logits.view(-1, logits.size(-1))
                    # num_classes = logits.size(-1)
                    # smoothing_value = self.label_smoothing / num_classes
                    # smoothed_targets = torch.full_like(flat_logits, smoothing_value).scatter_(
                    #     -1, labels.to(torch.long).view(-1, 1), 1.0 - self.label_smoothing + smoothing_value
                    # )
                    # loss_per_token = -log_probs_flat * smoothed_targets
                loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.to(torch.long).view(-1))
            else:
                # Shape: (batch_size,)
                loss = self.loss_fct_batch(logits.view(-1, logits.size(-1)), labels.to(torch.long).view(-1))
                # print(loss.size())
                # print(loss[0])
                # raise NotImplementedError
        elif self.training:
            raise ValueError("'labels' required during training")

        if not self.training:
            # Use beam search to generate a sequence of predicted tokens.

            # Shape: (batch_size, 1)
            initial_decoder_ids = torch.tensor(
                [[self.decoder_start_token_id]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).repeat(input_ids.shape[0], 1)

            initial_state = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_outputs.last_hidden_state,
                "encoder_attention_mask": attention_mask,
            }

            # Run the beam search.
            # Shape (predictions): (batch_size, beam_size, max_decoding_steps)
            # Shape (predicted_log_probs):   (batch_size, beam_size)
            predictions, predicted_log_probs = self.beam_search.search(
                initial_decoder_ids, initial_state, self.take_search_step
            )

        return T5Output(
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_all_hidden_states=encoder_outputs.all_hidden_states,
            decoder_last_hidden_state=(
                None if decoder_outputs is None else decoder_outputs.last_hidden_state
            ),
            decoder_all_hidden_states=(
                None if decoder_outputs is None else decoder_outputs.all_hidden_states
            ),
            encoder_attentions=encoder_outputs.attentions,
            decoder_attentions=None if decoder_outputs is None else decoder_outputs.attentions,
            cross_attentions=None if decoder_outputs is None else decoder_outputs.cross_attentions,
            loss=loss,
            logits=logits,
            predictions=predictions,
            predicted_log_probs=predicted_log_probs,
        )

    def forward_sequence(self,
        input_ids: IntT,
        attention_mask: Optional[BoolT] = None,
        labels: Optional[IntT] = None,
        decoder_attention_mask: Optional[BoolT] = None,):

        if attention_mask is None:
            attention_mask = ~(input_ids == self.pad_token_id)

        # Encode inputs.
        encoder_outputs: T5StackOutput = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            output_all_hidden_states=self.output_all_hidden_states,
        )

        logits: Optional[FloatT] = None
        loss: Optional[FloatT] = None
        decoder_outputs: Optional[T5StackOutput] = None
        predictions: Optional[IntT] = None
        predicted_log_probs: Optional[FloatT] = None

        if labels is not None:
            # Calculate loss against targets.

            if decoder_attention_mask is None:
                decoder_attention_mask = ~(labels == self.pad_token_id)

            # Get decoder inputs from shifting lm labels to the right and pre-pending
            # the decoder start token ID.
            # Shape (both): (batch_size, target_length)
            decoder_input_ids = self._shift_right(labels, self.decoder_start_token_id)

            # Replace possible -100 values in labels by `pad_token_id`
            decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id)

            # Decode.
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                output_attentions=self.output_attentions,
                output_all_hidden_states=self.output_all_hidden_states,
            )

            # Shape: (batch_size, target_length, vocab_size)
            logits = self._get_lm_logits(decoder_outputs.last_hidden_state)  # type: ignore[union-attr]

            # Shape: (batch_size, target_length)
            probs = torch.softmax(logits, dim=2)
            pred_confidence, pred_indices = torch.max(probs, dim=2)
            return pred_confidence, pred_indices

    def take_search_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        This function is what gets passed to the `BeamSearch.search` method. It takes
        predictions from the last timestep and the current state and outputs
        the log probabilities assigned to tokens for the next timestep, as well as the updated
        state.
        """
        decoder_cache: Optional[List[KeyValueStates]] = None
        decoder_cache_dict = {
            k: state[k].contiguous() for k in state if k.startswith("decoder_cache_")
        }
        if decoder_cache_dict:
            decoder_cache = self._dict_to_decoder_cache(decoder_cache_dict)

        if len(last_predictions.shape) == 1:
            last_predictions = last_predictions.unsqueeze(-1)

        decoder_outputs: T5StackOutput = self.decoder(
            input_ids=last_predictions,
            past_key_values=decoder_cache,
            encoder_hidden_states=state["encoder_hidden_states"],
            encoder_attention_mask=state["encoder_attention_mask"],
            use_cache=True,
        )

        # Shape: (group_size, 2, vocab_size)
        lm_logits = self._get_lm_logits(decoder_outputs.last_hidden_state)

        # Shape: (group_size, vocab_size)
        logits = lm_logits[:, -1, :]

        # Shape: (group_size, vocab_size)
        log_probabilities = F.log_softmax(logits, dim=-1)

        # Update state with decoder cache.
        decoder_cache = decoder_outputs.past_key_values
        assert decoder_cache is not None
        decoder_cache_dict = self._decoder_cache_to_dict(decoder_cache)
        state.update(decoder_cache_dict)

        return log_probabilities, state

    @staticmethod
    def _decoder_cache_to_dict(decoder_cache: List[KeyValueStates]) -> Dict[str, torch.Tensor]:
        cache_dict = {}
        for layer_index, layer_cache in enumerate(decoder_cache):
            # Each layer caches the key and value tensors for its self-attention and cross-attention.
            # Hence the `layer_cache` tuple has 4 elements.
            assert len(layer_cache) == 4
            for tensor_index, tensor in enumerate(layer_cache):
                key = f"decoder_cache_{layer_index}_{tensor_index}"
                cache_dict[key] = tensor
        return cache_dict

    def _dict_to_decoder_cache(self, cache_dict: Dict[str, torch.Tensor]) -> List[KeyValueStates]:
        decoder_cache: List[KeyValueStates] = []
        for block_index in range(self.decoder.num_blocks):
            base_key = f"decoder_cache_{block_index}_"
            layer_cache = (
                cache_dict[base_key + "0"].contiguous(),
                cache_dict[base_key + "1"].contiguous(),
                cache_dict[base_key + "2"].contiguous(),
                cache_dict[base_key + "3"].contiguous(),
            )
            decoder_cache.append(layer_cache)
        return decoder_cache
