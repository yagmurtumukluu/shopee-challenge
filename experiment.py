import logging

import numpy as np
import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
import record_keeper
import sentencepiece as spm
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from config import SPIECE_MODEL_FILE

sp = spm.SentencePieceProcessor()
logging.info(sp.load(SPIECE_MODEL_FILE))
