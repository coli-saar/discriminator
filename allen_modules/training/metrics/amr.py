from amrlib.models.parse_t5.penman_serializer import PenmanDeSerializer
from amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch
from amrlib.utils.logging import silence_penman
from allennlp.training.metrics.metric import Metric
from typing import List, Dict, Any

@Metric.register("amr")
class AMRMetrics(Metric):
    def __init__(self, print_err=True):
        self.match_num = 0
        self.total_num = 0
        # self.skip_num = 0
        self.print_err = print_err
        self.pred_entries = []
        self.gold_entries = []

    def reset(self) -> None:
        self.match_num = 0
        self.total_num = 0
        self.pred_entries = []
        self.gold_entries = []

    def __call__(self, predicted_text: List[str],
                        metadata: List[Dict]):
        silence_penman()
        for i in range(len(predicted_text)):
            predstr = predicted_text[i]
            goldstr = metadata[i]['target_text']
            gstring = PenmanDeSerializer(predstr).get_graph_string()
            if gstring is not None:
                predstr = gstring
            else:
                print("ERROR: "+predstr)

            gold_gstring = PenmanDeSerializer(goldstr).get_graph_string()
            if gold_gstring is not None:
                goldstr = gold_gstring
            else:
                raise AssertionError

            if predstr != goldstr and  self.print_err:
                print("#######DEBUG##########")
                print("Smatch PRED: "+repr(predstr))
                print("Smatch GOLD: "+repr(goldstr))

            self.pred_entries.append(predstr)
            self.gold_entries.append(goldstr)

            if predstr == goldstr:
                self.match_num += 1
            self.total_num += 1

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        acc = self.match_num * 1.0 / self.total_num if self.total_num != 0 else 0
        metric_dict = {
            'deserial_acc': acc,
        }
        if reset:
            precision, recall, f_score = compute_smatch(self.pred_entries, self.gold_entries)
            smatch_dict = {
                'deserial_precision': precision,
                'deserial_recall': recall,
                'deserial_f_score': f_score
            }
            metric_dict.update(smatch_dict)
            self.reset()
        return metric_dict
