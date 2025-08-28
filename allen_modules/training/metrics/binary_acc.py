from allennlp.training.metrics.metric import Metric

@Metric.register("binary_acc")
class BinaryAccuracy(Metric):
    def __init__(self) -> None:
        self.corr = 0
        self.num = 0

    def __call__(self, predictions, labels):
        for x, y in zip(predictions, labels):
            if x == y:
                self.corr += 1
            self.num += 1

    def get_metric(self, reset: bool = False):
        result = {"binary_acc": self.corr * 1.0 /self.num}
        if reset:
            self.reset()
        return result

    def reset(self):
        self.corr = 0
        self.num = 0