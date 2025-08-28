from allennlp.training.metrics.metric import Metric
from typing import List, Dict, Any

@Metric.register("ece")
class ExpectedCalibrationError(Metric):
    def __init__(self, interval=0.2):
        self.interval = interval
        self.bin_num = int(1.0 // interval + 1)
        self.conf_bins = {min(round(interval*i, 2), 1):[] for i in range(1,self.bin_num+1)}
        self.typedict = {}

    def reset(self) -> None:
        self.conf_bins = {k:[] for k in self.conf_bins}
        self.typedict = {}

    def __call__(self, predicted_text: List[str],
                        metadata: List[Dict],
                        confidence: List[float]):
        for i in range(len(predicted_text)):
            predstr = predicted_text[i]
            goldstr = metadata[i]['target_text']

            conf_val = confidence[i]
            acc_val = 1 if predstr == goldstr else 0
            conf_key = min(round((int(conf_val // self.interval)+1) * self.interval, 2), 1)
            self.conf_bins[conf_key].append((acc_val, conf_val))
            if "gen_type" in metadata[i]:
                gen_type = metadata[i]["gen_type"]
                if gen_type not in self.typedict:
                    self.typedict[gen_type] = {min(round(self.interval*i, 2), 1):[] for i in range(1,self.bin_num+1)}
                self.typedict[gen_type][conf_key].append((acc_val, conf_val))

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        metric_dict = {"overall_ece": 0, "overall_confidence":0, "overall_accuracy":0}
        num_instances = sum([len(self.conf_bins[k]) for k in self.conf_bins.keys()])
        for conf_key in self.conf_bins:
            tuple_list = self.conf_bins[conf_key]
            bin_size = len(tuple_list)
            avg_acc = sum([x[0] for x in tuple_list]) / bin_size if bin_size > 0 else 0
            avg_conf = sum([x[1] for x in tuple_list]) / bin_size if bin_size > 0 else 0
            bin_ece = abs(avg_acc-avg_conf)
            bin_weight = bin_size*1.0 / num_instances * bin_ece
            metric_dict["overall_conf_{}_ece".format(conf_key)] = bin_ece
            metric_dict["overall_conf_{}_confidence".format(conf_key)] = avg_conf
            metric_dict["overall_conf_{}_accuracy".format(conf_key)] = avg_acc
            metric_dict["overall_conf_{}_size".format(conf_key)] = bin_size
            metric_dict["overall_ece"] += bin_weight
            metric_dict["overall_confidence"] += avg_conf * bin_size / num_instances
            metric_dict["overall_accuracy"] += avg_acc * bin_size / num_instances
        # print(metric_dict)
        if self.typedict:
            for gen_type in self.typedict:
                metric_dict["{}_ece".format(gen_type)] = 0
                metric_dict["{}_confidence".format(gen_type)] = 0
                metric_dict["{}_accuracy".format(gen_type)] = 0
                num_instances = sum([len(self.typedict[gen_type][k]) for k in self.typedict[gen_type].keys()])
                for conf_key in self.typedict[gen_type]:
                    tuple_list = self.typedict[gen_type][conf_key]
                    bin_size = len(tuple_list)
                    avg_acc = sum([x[0] for x in tuple_list]) / bin_size if bin_size > 0 else 0
                    avg_conf = sum([x[1] for x in tuple_list]) / bin_size if bin_size > 0 else 0
                    bin_ece = abs(avg_acc - avg_conf)
                    bin_weight = bin_size * 1.0 / num_instances * bin_ece
                    metric_dict["{}_conf_{}_ece".format(gen_type, conf_key)] = bin_ece
                    metric_dict["{}_ece".format(gen_type)] += bin_weight
                    metric_dict["{}_conf_{}_confidence".format(gen_type,conf_key)] = avg_conf
                    metric_dict["{}_conf_{}_accuracy".format(gen_type,conf_key)] = avg_acc
                    metric_dict["{}_conf_{}_size".format(gen_type, conf_key)] = bin_size
                    metric_dict["{}_confidence".format(gen_type)] += avg_conf * bin_size / num_instances
                    metric_dict["{}_accuracy".format(gen_type)] += avg_acc * bin_size / num_instances
        # print(metric_dict)
        # raise NotImplementedError
        if reset:
            self.reset()
        return metric_dict



@Metric.register("token_ece")
class TokenExpectedCalibrationError(Metric):
    def __init__(self, interval=0.2):
        self.interval = interval
        self.bin_num = int(1.0 // interval + 1)
        self.conf_bins = {min(round(interval*i, 2), 1):[] for i in range(1,self.bin_num+1)}
        self.typedict = {}

    def reset(self) -> None:
        self.conf_bins = {k:[] for k in self.conf_bins}
        self.typedict = {}

    def __call__(self, confidence: List,
                        accuracy: List,
                        metadata: List[Dict],
                        ):
        for i in range(len(confidence)):
            for j in range(len(confidence[i])):
                conf_val = confidence[i][j]
                acc_val = accuracy[i][j]
                if conf_val == -1:
                    assert acc_val == -1
                    continue
                conf_key = min(round((int(conf_val // self.interval)+1) * self.interval, 2), 1)
                self.conf_bins[conf_key].append((acc_val, conf_val))
                if "gen_type" in metadata[i]:
                    gen_type = metadata[i]["gen_type"]
                    if gen_type not in self.typedict:
                        self.typedict[gen_type] = {min(round(self.interval * i, 2), 1): [] for i in
                                                   range(1, self.bin_num + 1)}
                    self.typedict[gen_type][conf_key].append((acc_val, conf_val))

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        metric_dict = {"overall_token_ece": 0,"overall_token_confidence":0, "overall_token_accuracy":0}
        num_instances = sum([len(self.conf_bins[k]) for k in self.conf_bins.keys()])
        for conf_key in self.conf_bins:
            tuple_list = self.conf_bins[conf_key]
            bin_size = len(tuple_list)
            avg_acc = sum([x[0] for x in tuple_list]) / bin_size if bin_size > 0 else 0
            avg_conf = sum([x[1] for x in tuple_list]) / bin_size if bin_size > 0 else 0
            bin_ece = abs(avg_acc-avg_conf)
            bin_weight = bin_size*1.0 / num_instances * bin_ece
            metric_dict["overall_token_conf_{}_ece".format(conf_key)] = bin_ece
            metric_dict["overall_token_conf_{}_size".format(conf_key)] = bin_size
            metric_dict["overall_token_conf_{}_confidence".format(conf_key)] = avg_conf
            metric_dict["overall_token_conf_{}_accuracy".format(conf_key)] = avg_acc
            metric_dict["overall_token_ece"] += bin_weight
            metric_dict["overall_token_confidence"] += avg_conf * bin_size / num_instances
            metric_dict["overall_token_accuracy"] += avg_acc * bin_size / num_instances
        # print(metric_dict)
        if self.typedict:
            for gen_type in self.typedict:
                metric_dict["{}_token_ece".format(gen_type)] = 0
                metric_dict["{}_token_confidence".format(gen_type)] = 0
                metric_dict["{}_token_accuracy".format(gen_type)] = 0
                num_instances = sum([len(self.typedict[gen_type][k]) for k in self.typedict[gen_type].keys()])
                for conf_key in self.typedict[gen_type]:
                    tuple_list = self.typedict[gen_type][conf_key]
                    bin_size = len(tuple_list)
                    avg_acc = sum([x[0] for x in tuple_list]) / bin_size if bin_size > 0 else 0
                    avg_conf = sum([x[1] for x in tuple_list]) / bin_size if bin_size > 0 else 0
                    bin_ece = abs(avg_acc - avg_conf)
                    bin_weight = bin_size * 1.0 / num_instances * bin_ece
                    metric_dict["{}_token_conf_{}_ece".format(gen_type, conf_key)] = bin_ece
                    metric_dict["{}_token_ece".format(gen_type)] += bin_weight
                    metric_dict["{}_token_conf_{}_confidence".format(gen_type,conf_key)] = avg_conf
                    metric_dict["{}_token_conf_{}_accuracy".format(gen_type,conf_key)] = avg_acc
                    metric_dict["{}_token_conf_{}_size".format(gen_type, conf_key)] = bin_size
                    metric_dict["{}_token_confidence".format(gen_type)] += avg_conf * bin_size / num_instances
                    metric_dict["{}_token_accuracy".format(gen_type)] += avg_acc * bin_size / num_instances
        # print(metric_dict)
        # raise NotImplementedError
        if reset:
            self.reset()
        return metric_dict