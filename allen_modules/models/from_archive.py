"""
`Model` is an abstract class representing
an AllenNLP model.
"""

import logging
import os
from os import PathLike
import re
from typing import Dict, List, Set, Type, Optional, Union

import numpy
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params, remove_keys_from_params
from allennlp.common.registrable import Registrable
from allennlp.data import Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.nn import util
from allennlp.nn.module import Module
from allennlp.nn.parallel import DdpAccelerator
from allennlp.nn.regularizers import RegularizerApplicator

logger = logging.getLogger(__name__)

# When training a model, many sets of weights are saved. By default we want to
# save/load this set of weights.
_DEFAULT_WEIGHTS = "best.th"


class Model(Module, Registrable):
    """
    This abstract class represents a model to be trained. Rather than relying completely
    on the Pytorch Module, we modify the output spec of `forward` to be a dictionary.

    Models built using this API are still compatible with other pytorch models and can
    be used naturally as modules within other models - outputs are dictionaries, which
    can be unpacked and passed into other layers. One caveat to this is that if you
    wish to use an AllenNLP model inside a Container (such as nn.Sequential), you must
    interleave the models with a wrapper module which unpacks the dictionary into
    a list of tensors.

    In order for your model to be trained using the [`Trainer`](../training/trainer.md)
    api, the output dictionary of your Model must include a "loss" key, which will be
    optimised during the training process.

    Finally, you can optionally implement :func:`Model.get_metrics` in order to make use
    of early stopping and best-model serialization based on a validation metric in
    `Trainer`. Metrics that begin with "_" will not be logged
    to the progress bar by `Trainer`.

    The `from_archive` method on this class is registered as a `Model` with name "from_archive".
    So, if you are using a configuration file, you can specify a model as `{"type": "from_archive",
    "archive_file": "/path/to/archive.tar.gz"}`, which will pull out the model from the given
    location and return it.

    # Parameters

    vocab: `Vocabulary`
        There are two typical use-cases for the `Vocabulary` in a `Model`: getting vocabulary sizes
        when constructing embedding matrices or output classifiers (as the vocabulary holds the
        number of classes in your output, also), and translating model output into human-readable
        form.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "model", it gets specified as a top-level parameter, then is passed in to the model
        automatically.

    regularizer: `RegularizerApplicator`, optional
        If given, the `Trainer` will use this to regularize model parameters.

    serialization_dir: `str`, optional
        The directory in which the training output is saved to, or the directory the model is loaded from.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "model".

    ddp_accelerator : `Optional[DdpAccelerator]`, optional
        The :class:`allennlp.nn.parallel.ddp_accelerator.DdpAccelerator` used in distributing training.
        If not in distributed training, this will be `None`.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "model", it gets specified as "ddp_accelerator" in the "distributed" part of the configs, and is then
        passed in to the model automatically.

        It will be available to `Model` instances as `self.ddp_accelerator`.
    """

    @classmethod
    def from_archive(cls, archive_file: str, vocab: Vocabulary = None, weights_file:str = None) -> "Model":
        """
        Loads a model from an archive file.  This basically just calls
        `return archival.load_archive(archive_file).model`.  It exists as a method here for
        convenience, and so that we can register it for easy use for fine tuning an existing model
        from a configs file.

        If `vocab` is given, we will extend the loaded model's vocabulary using the passed vocab
        object (including calling `extend_embedder_vocab`, which extends embedding layers).
        """
        from allennlp.models.archival import load_archive  # here to avoid circular imports

        model = load_archive(archive_file, weights_file=weights_file).model
        if vocab:
            model.vocab.extend_from_vocab(vocab)
            model.extend_embedder_vocab()
        return model

    # @classmethod
    # def from_archive(cls, archive_file: str, vocab: Vocabulary = None, weights_file: str = None) -> "Model":
    #     """
    #     Loads a model from an archive file.  This basically just calls
    #     `return archival.load_archive(archive_file).model`.  It exists as a method here for
    #     convenience, and so that we can register it for easy use for fine tuning an existing model
    #     from a configs file.
    #
    #     If `vocab` is given, we will extend the loaded model's vocabulary using the passed vocab
    #     object (including calling `extend_embedder_vocab`, which extends embedding layers).
    #     """
    #     from allennlp.models.archival import load_archive  # here to avoid circular imports
    #
    #     model = load_archive(archive_file, weights_file=weights_file).model
    #     if vocab:
    #         model.vocab.extend_from_vocab(vocab)
    #         model.extend_embedder_vocab()
    #     return model


# We can't decorate `Model` with `Model.register()`, because `Model` hasn't been defined yet.  So we
# put this down here.
Model.register("from_archive", constructor="from_archive")(Model)
# Model.register("from_archive", constructor="from_archive")(Model)


def remove_weights_related_keys_from_params(
    params: Params, keys: List[str] = ["pretrained_file", "initializer"]
):
    remove_keys_from_params(params, keys)


def remove_pretrained_embedding_params(params: Params):
    """This function only exists for backwards compatibility.
    Please use `remove_weights_related_keys_from_params()` instead."""
    remove_keys_from_params(params, ["pretrained_file"])