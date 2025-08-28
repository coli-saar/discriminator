from typing import Iterable, Optional

import torch

from allennlp.common.registrable import Registrable


class Logger(Registrable):

    def __call__(
        self, *args, **kwargs
    ):
        raise NotImplementedError
