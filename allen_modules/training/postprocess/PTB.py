from typing import Union, Dict, Any, Optional, List
import re

from allen_modules.training.postprocess.postprocessor import Postprocessor
from allen_modules.training.postprocess.simple import SimplePostprocessor
@Postprocessor.register("ptb")
class PTBPostprocessor(SimplePostprocessor):

    def __init__(self, segment_symbols: List[str] = None, skip_special_tokens=True):
        """
        :param segment_symbols: symbol strings that will be segmented

        """
        super(PTBPostprocessor, self).__init__(segment_symbols, skip_special_tokens)
        self.seg_symbols = []

    def __call__(self, predicted_texts: List[str]):
        """

        :param predicted_texts: text string list returned by allennlp models
        """
        predicted_texts = self.format(predicted_texts, segment_symbols=self.seg_symbols)

        return [self.postprocess_mr(predicted_text) for predicted_text in predicted_texts]

    def close_brackets(self, predicted_texts: str):
        tokens = predicted_texts.split()
        lb = 0
        for token in tokens:
            if token == "(":
                lb += 1
            if token == ")":
                lb -= 1
        if lb >= 1:
            for i in range(lb):
                tokens.append(")")
        elif lb <= -1:
            for i in range(-lb):
                tokens.insert(0, "(")
        string = " ".join(tokens)
        string = string.replace("( )", "")
        string = " ".join(string.split())
        return string

    def postprocess_mr(self, predicted_texts: str):

        predicted_texts = self.close_brackets(predicted_texts)

        return predicted_texts