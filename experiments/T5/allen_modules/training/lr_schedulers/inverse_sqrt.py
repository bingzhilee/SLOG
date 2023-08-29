import torch

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("inverse_sqrt")
class InverseSqrtLR(LearningRateScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        warmup_end_lr: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.lr = warmup_end_lr
        self.decay_factor = warmup_end_lr * warmup_steps ** 0.5
        super().__init__(optimizer, last_epoch=last_epoch)

    def step(self, metric: float = None) -> None:
        pass

    def step_batch(self, batch_num_total: int = None) -> None:
        if batch_num_total is None:
            self.last_epoch += 1  # type: ignore
        else:
            self.last_epoch = batch_num_total
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_values()):
            param_group["lr"] = learning_rate

    def get_values(self):
        step = max(self.last_epoch, 1)
        if step <= self.warmup_steps:
            scale = step / self.warmup_steps * self.lr
        else:
            scale =  self.decay_factor * step ** -0.5

        return [scale for _ in range(len(self.base_values))]