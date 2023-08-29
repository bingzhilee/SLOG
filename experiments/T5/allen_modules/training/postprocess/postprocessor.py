"""
    We use this base class to implement task-specific post-process functions
"""

from typing import Union, Dict, Any, Optional, List
from os import PathLike
from pathlib import Path
import torch
import logging

from allennlp.common.checks import check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, int_to_device
from allennlp.nn import util as nn_util
from allennlp.common import Registrable
from allennlp.models import Model
from allennlp.data import DataLoader

class Postprocessor(Registrable):
    """
        Base class for postprocessing predictions of allennlp models
    """

    def __call__(self, predicted_texts: List[str]):
        """

        :param predicted_texts: text string list returned by allennlp models
        """
        raise NotImplementedError

    def format(self):
        """

        :param predicted_texts: text string list returned by allennlp models
        """
        raise NotImplementedError