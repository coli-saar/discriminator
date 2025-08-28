from os import PathLike
from typing import Optional, Dict, Any, Union, List, Tuple

import numpy as np
import torch

from allennlp.common import Registrable
from allen_modules.evaluation.logger.logger import Logger
@Logger.register("T5_features")
class T5FeaturesLogger(Logger):
    def __init__(self) -> None:
        pass

    def __call__(self, encoder_input_ids, output_dict, metadata):
        # Save labels as list
        labels = [output_dict["predicted_text"][i] == metadata[i]["target_text"] for i in range(len(metadata))]

        features = {}

        # Save raw confidence as a list
        features["raw_confidence"] = output_dict["predicted_probs"]

        # Save length of source and target as a list

        features["source_length"] = self.get_true_input_length(encoder_input_ids)
        features["target_length"] = self.get_true_output_length(output_dict["predictions"])

        # Feature: CrossAttention Entropy of each heads averaged over tokens
        cross_attentions = output_dict["cross_attentions"]
        # print(cross_attentions[0].size())
        # raise NotImplementedError
        cross_attentions = cross_attentions[-1]
        # Assume shape (bsz, head_num, dec_len, enc_len)
        # Entropy shape: (bsz, head_num, dec_len)
        attention_mask = (cross_attentions == 0).bool()
        masked_cross_attentions = cross_attentions.masked_fill(attention_mask, 1e-9)
        entropy = torch.sum(-cross_attentions * torch.log2(masked_cross_attentions), dim=-1)
        # Shape: (bsz, head_num)
        avg_entropy = torch.mean(entropy, dim=-1).tolist()
        features["cross_attention_entropy"] = avg_entropy

        output_dict["logger_output"] = [None] * len(metadata)
        for i in range(len(metadata)):
            output_dict["logger_output"][i] = {}
            # output_dict["logger_output"][i]["metadata"] = metadata[i]
            output_dict["logger_output"][i]["source_texts"] = metadata[i]["source_text"]
            output_dict["logger_output"][i]["boolean_labels"] = labels[i]
            output_dict["logger_output"][i]["predicted_texts"] = output_dict["predicted_text"][i]
            output_dict["logger_output"][i]["gold_texts"] = metadata[i]["target_text"]
            features_dict = {}

            for feature in features:
                # print(features[feature])
                features_dict[feature] = features[feature][i]

            output_dict["logger_output"][i]["features"] = features_dict

        return output_dict

    def get_true_input_length(self, encoder_input_ids):
        encoder_input_ids = encoder_input_ids.tolist()
        padding_token = 0  # assuming that 0 is used as the padding token
        output = [None] * len(encoder_input_ids)
        for i in range(len(encoder_input_ids)):
            if padding_token in encoder_input_ids[i]:
                padding_positions = encoder_input_ids[i].index(padding_token)
            else:
                padding_positions = len(encoder_input_ids[i])
            output[i] = padding_positions
        return output

    def get_true_output_length(self, decoder_input_ids):
        decoder_input_ids = decoder_input_ids.tolist()
        padding_token = 1
        output = [None] * len(decoder_input_ids)
        for i in range(len(decoder_input_ids)):
            if padding_token in decoder_input_ids[i]:
                padding_positions = decoder_input_ids[i].index(padding_token)
            else:
                padding_positions = len(decoder_input_ids[i])
            output[i] = padding_positions
        return output