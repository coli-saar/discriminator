from allennlp.training.metrics.metric import Metric
from typing import List, Dict, Any

@Metric.register("tokens_acc")
class TokenMatchAcc(Metric):
    def __init__(self, ignore_curly_brackets=True):
        self.match_num = 0
        self.total_num = 0

    def reset(self) -> None:
        self.match_num = 0
        self.total_num = 0

    def __call__(self, predicted_text: List[str],
                        metadata: List[Dict]):
        for i in range(len(predicted_text)):
            pred_tokens = predicted_text[i].split()
            gold_tokens = metadata[i]['target_text'].split()
            for j in range(min(len(pred_tokens), len(gold_tokens))):
                if pred_tokens[j] == gold_tokens[j]:
                    self.match_num += 1
                self.total_num += 1

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        acc = self.match_num * 1.0 / self.total_num if self.total_num != 0 else 0
        metric_dict = {'tokens_acc': acc}

        if reset:
            self.reset()
        return metric_dict