import copy
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import BertTokenizerFast, DistilBertTokenizerFast

#from data_augmentation.randaugment import FIX_MATCH_AUGMENTATION_POOL, RandAugment


_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]


def initialize_transform(
    transform_name, config, dataset, is_training, additional_transform_name=None
):
    """
    By default, transforms should take in `x` and return `transformed_x`.
    For transforms that take in `(x, y)` and return `(transformed_x, transformed_y)`,
    set `do_transform_y` to True when initializing the WILDSSubset.
    """
    if transform_name is None:
        return None
    elif transform_name == "bert":
        return initialize_bert_transform(config)

def initialize_bert_transform(config):
    def get_bert_tokenizer(model):
        if model == "bert-base-uncased":
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")

    assert "bert" in config.model
    assert config.max_token_length is not None

    tokenizer = get_bert_tokenizer(config.model)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_token_length,
            return_tensors="pt",
        )
        if config.model == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif config.model == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform

class MultipleTransforms(object):
    """When multiple transformations of the same data need to be returned."""

    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x):
        return tuple(transform(x) for transform in self.transformations)
