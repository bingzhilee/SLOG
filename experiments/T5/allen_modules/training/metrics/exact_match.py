from allennlp.training.metrics.metric import Metric
from typing import List, Dict, Any

@Metric.register("acc")
class ExactMatchAcc(Metric):
    def __init__(self, print_err=False):
        self.match_num = 0
        self.total_num = 0
        self.typedict = {}
        self.print_err = print_err

    def reset(self) -> None:
        self.match_num = 0
        self.total_num = 0
        self.typedict = {}

    def __call__(self, predicted_text: List[str],
                        metadata: List[Dict]):
        for i in range(len(predicted_text)):
            predstr = predicted_text[i]
            goldstr = metadata[i]['target_text']

            # if i == 0:
            #     print("######PREDICTION#####")
            #     print(predstr)
            #     print(goldstr)

            if predstr != goldstr and  self.print_err:
                print("#######DEBUG##########")
                print("PRED: "+repr(predstr))
                print("GOLD: "+repr(goldstr))


            if predstr == goldstr:
                self.match_num += 1
                if "gen_type" in metadata[i]:
                    gen_type = metadata[i]["gen_type"]
                    if gen_type not in self.typedict:
                        self.typedict[gen_type] = [0, 0]
                    self.typedict[gen_type][0] += 1

            self.total_num += 1
            if "gen_type" in metadata[i]:
                gen_type = metadata[i]["gen_type"]
                if gen_type not in self.typedict:
                    self.typedict[gen_type] = [0, 0]
                self.typedict[gen_type][1] += 1

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        acc = self.match_num * 1.0 / self.total_num if self.total_num != 0 else 0
        metric_dict = {'acc': acc}
        if self.typedict:
            for gen_type in self.typedict:
                metric_dict[gen_type] = self.typedict[gen_type][0]*1.0 / self.typedict[gen_type][1]
        if reset:
            self.reset()
        return metric_dict