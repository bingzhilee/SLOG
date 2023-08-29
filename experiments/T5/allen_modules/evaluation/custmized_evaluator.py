"""
Evaluator class for evaluating a model with a given dataset
"""
from typing import Union, Dict, Any, Optional
from os import PathLike
from pathlib import Path
import torch
import logging
import os, pathlib

from allennlp.common.checks import check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, int_to_device
from allennlp.nn import util as nn_util
from allennlp.common import Registrable
from allennlp.models import Model
from allennlp.data import DataLoader
from allennlp.evaluation.evaluator import Evaluator
from allennlp.evaluation.serializers.serializers import Serializer, SimpleSerializer

logger = logging.getLogger(__name__)

@Evaluator.register("cust_evaluator")
class SimpleEvaluator(Evaluator):
    """
    Simple evaluator implementation. Uses the vanilla evaluation code.

    # Parameters

    batch_postprocessor: `Postprocessor`, optional (default=`SimplePostprocessor`)
        The postprocessor to use for turning both the batches and the outputs
        of the model into human readable data.

    cuda_device : `Union[int, torch.device]`, optional (default=`-1`)
        The cuda device to use for this evaluation.  The model is assumed to
         already be using this device; this parameter is only used for moving
         the input data to the correct device.

    postprocessor_fn_name: `str`, optional (default=`"make_output_human_readable"`)
        Function name of the model's postprocessing function.
    """

    def __init__(
        self,
        batch_serializer: Optional[Serializer] = None,
        cuda_device: Union[int, torch.device] = -1,
        postprocessor_fn_name: str = "make_output_human_readable",
    ):
        super(SimpleEvaluator, self).__init__(batch_serializer, cuda_device, postprocessor_fn_name)

    def __call__(
        self,
        model: Model,
        data_loader: DataLoader,
        batch_weight_key: str = None,
        metrics_output_file: Union[str, PathLike] = None,
        predictions_output_file: Union[str, PathLike] = None,
        log_probabilities: bool = False,
    ):
        """
        Evaluate a single data source.

        # Parameters

        model : `Model`
            The model to evaluate
        data_loader : `DataLoader`
            The `DataLoader` that will iterate over the evaluation data (data loaders already contain
            their data).
        batch_weight_key : `str`, optional (default=`None`)
            If given, this is a key in the output dictionary for each batch that specifies how to weight
            the loss for that batch.  If this is not given, we use a weight of 1 for every batch.
        metrics_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the final metrics to.
        predictions_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the predictions to.

        # Returns

        metrics: `Dict[str, Any]`
            The metrics from evaluating the file.
        """
        # print(model.trainer._callbacks)
        # raise NotImplementedError
        check_for_gpu(self.cuda_device)
        data_loader.set_target_device(int_to_device(self.cuda_device))
        metrics_output_file = Path(metrics_output_file) if metrics_output_file is not None else None
        if predictions_output_file is not None:
            pathlib.Path(os.path.dirname(predictions_output_file)).mkdir(exist_ok=True)
            predictions_file = Path(predictions_output_file).open("w", encoding="utf-8")
            readable_predictions_file = Path(predictions_output_file+".read.tsv").open("w", encoding="utf-8")
            mistake_predictions_file = Path(predictions_output_file+".err.tsv").open("w", encoding="utf-8")
            if log_probabilities:
                prob_predictions_file = Path(predictions_output_file+".prob.md").open("w", encoding="utf-8")
        else:
            predictions_file = None  # type: ignore

        model_postprocess_function = getattr(model, self.postprocessor_fn_name, None)

        with torch.no_grad():
            model.eval()

            iterator = iter(data_loader)
            logger.info("Iterating over dataset")
            generator_tqdm = Tqdm.tqdm(iterator)
            # Number of batches in instances.
            batch_count = 0
            # Number of batches where the model produces a loss.
            loss_count = 0
            # Cumulative weighted loss
            total_loss = 0.0
            # Cumulative weight across all batches.
            total_weight = 0.0

            if predictions_output_file is not None:
                mistake_data = []
                readable_data = []
                prob_data = []

            for batch in generator_tqdm:
                batch_count += 1
                batch = nn_util.move_to_device(batch, self.cuda_device)
                output_dict = model(**batch)
                loss = output_dict.get("loss")

                metrics = model.get_metrics()

                if loss is not None:
                    loss_count += 1
                    if batch_weight_key:
                        weight = output_dict[batch_weight_key].item()
                    else:
                        weight = 1.0

                    total_weight += weight
                    total_loss += loss.item() * weight
                    # Report the average loss so far.
                    metrics["loss"] = total_loss / total_weight

                description = (
                    ", ".join(
                        [
                            "%s: %.2f" % (name, value)
                            for name, value in metrics.items()
                            if not name.startswith("_")
                        ]
                    )
                    + " ||"
                )
                generator_tqdm.set_description(description, refresh=False)

                # TODO(gabeorlanski): Add in postprocessing the batch for token
                #  metrics
                # print(dict(filter(lambda x: x[0] in ["predictions", "loss"], output_dict.items)))
                # raise NotImplementedError
                if predictions_file is not None:
                    save_keys = ["predictions", "loss"]
                    if "logger_output" in output_dict:
                        save_keys.append("logger_output")
                    predictions_file.write(
                        self.batch_serializer(
                            batch,
                            dict(filter(lambda x: x[0] in save_keys, output_dict.items())),
                            # output_dict,
                            data_loader,
                            output_postprocess_function=model_postprocess_function,
                        )
                        + "\n"
                    )
                    # print(output_dict["predicted_text"][0])
                    # print(len(batch["metadata"]))
                    # print(batch["metadata"][0]["target_text"])
                    # print(batch["metadata"][0]["source_text"])
                    ignore_curly_brackets = True
                    log_prob_line_num = 100

                    if "predicted_text" in output_dict:
                        for idx in range(len(output_dict["predicted_text"])):
                            gold = batch["metadata"][idx]["target_text"]
                            # print(gold)
                            tok_pred = output_dict["predicted_text"][idx]
                            line = "{}\t{}\t{}\n".format(batch["metadata"][idx]["source_text"], \
                                                         gold, \
                                                         output_dict["predicted_text"][idx])
                            if gold != tok_pred:
                                mistake_data.append(line)
                            readable_data.append(line)

                            if log_probabilities:
                                if len(prob_data) >= log_prob_line_num:
                                    continue
                                pred_probs = output_dict["predicted_probs"][idx]
                                # print(output_dict["loss"])
                                # print(output_dict["gold_probs"])
                                gold_probs = output_dict["gold_probs"][idx]
                                if "incorr_predicted_text" in output_dict:
                                    if output_dict["incorr_predicted_text"][idx] is not None:
                                        incorr_predicted_text = "##### PRED:\t{}".format(output_dict["incorr_predicted_text"][idx])
                                        incorr_gold_text = "##### GOLD:\t{}".format(output_dict["incorr_gold_text"][idx])
                                        incorr_confidence = "##### Confidence:\t ${}$".format(output_dict["incorr_confidence"][idx])

                                        line = "{}\n{}\n{}\n------\n".format(incorr_gold_text, incorr_predicted_text, incorr_confidence)

                                # line = "{}\n{}\n{}\n{}\n{}\n\n".format(batch["metadata"][idx]["source_text"],
                                #                                        gold, gold_probs,
                                #                                        output_dict["predicted_text"][idx], pred_probs)
                                        prob_data.append(line)
                    else:
                        for idx in range(len(output_dict["predicted_tokens"])):
                            tok_pred = output_dict["predicted_tokens"][idx]
                            line = "{}\t{}\t{}\n".format(batch["metadata"][idx]["source_text"], \
                                                         batch["metadata"][idx]["target_tokens"], \
                                                         output_dict["predicted_tokens"][idx])
                            if tok_pred != batch["metadata"][idx]["target_tokens"]:
                                mistake_data.append(line)
                            readable_data.append(line)


            if predictions_file is not None:
                predictions_file.close()
                readable_predictions_file.writelines(readable_data)
                readable_predictions_file.close()
                mistake_predictions_file.writelines(mistake_data)
                mistake_predictions_file.close()
                if log_probabilities:
                    prob_predictions_file.writelines(prob_data)
                    prob_predictions_file.close()
            final_metrics = model.get_metrics(reset=True)
            if loss_count > 0:
                # Sanity check
                if loss_count != batch_count:
                    raise RuntimeError(
                        "The model you are trying to evaluate only sometimes produced a loss!"
                    )
                final_metrics["loss"] = total_loss / total_weight

            if metrics_output_file is not None:
                dump_metrics(str(metrics_output_file), final_metrics, log=True)

            return final_metrics

    def _to_params(self) -> Dict[str, Any]:
        return {
            "type": "simple",
            "cuda_device": self.cuda_device,
            "batch_postprocessor": self.batch_serializer.to_params(),
        }