

class ExactMatchAcc:
    def __init__(self):
        self.results = {}
        self.gold_num = 0
        self.corr_num = 0

    def add_batch(self, pred, gold, gen_types=None):
        for i in range(len(pred)):
            if pred[i] == gold[i]:
                self.corr_num += 1
            else:
                print("PRED: {}".format(pred[i]))
                print("GOLD: {}".format(gold[i]))

            if gen_types is not None:
                gen_type = gen_types[i]
                if gen_type not in self.results:
                    self.results[gen_type] = [0, 0]
                if pred[i] == gold[i]:
                    self.results[gen_type][0] += 1
                self.results[gen_type][1] += 1

            self.gold_num += 1
        print(self.compute_metric())


    def compute_metric(self):
        metric_dict = {}
        metric_dict["ACC"] = self.corr_num * 1.0 / self.gold_num
        for gen_type in self.results:
            metric_dict[gen_type] = self.results[gen_type][0] * 1.0 / self.results[gen_type][1]
        return metric_dict