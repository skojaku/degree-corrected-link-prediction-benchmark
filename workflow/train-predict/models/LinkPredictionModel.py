# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-07-03 13:24:24
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-03 13:43:04
from .EmbeddingModels import *
from .StackingModel import *
from .SEALModel import *
from .NetworkModels import *

link_prediction_models = {
    "embedding": EmbeddingLinkPredictor,
    "seal": SEALLinkPredictor,
    "stacklp": StackingLinkPredictor,
    "network": NetworkLinkPredictor,
}
