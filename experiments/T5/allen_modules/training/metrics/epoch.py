from allennlp.training.metrics.metric import Metric

@Metric.register("last_epoch")
class EpochsPassed(Metric):
    def __init__(self) -> None:
        self.epochs_passed = 0
        self.this_epoch = False

    def __call__(self, *args, **kwargs):
        if not self.this_epoch:
            self.epochs_passed += 1
        self.this_epoch = True

    def get_metric(self, reset: bool = False):
        result = {"epochs": self.epochs_passed}
        if reset:
            self.reset()
        return result

    def reset(self):
        self.this_epoch = False