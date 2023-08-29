import os

import yaml
from transformers import Seq2SeqTrainer
from transformers import get_polynomial_decay_schedule_with_warmup


class TangSynTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        training_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.training_config = training_config

    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Setup the scheduler. The optimizer of the trainer must have been
        set up either before this method is called or passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.args.lr_scheduler_type == "polynomial":
            if self.lr_scheduler is None:

                print("Using custom polynomial scheduler.")
                self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(
                        num_training_steps),
                    num_training_steps=num_training_steps,
                    power=1.0,
                    lr_end=5e-6
                )

            return self.lr_scheduler

        return super().create_scheduler(num_training_steps, optimizer)

    def save_model(self, output_dir=None, _internal_call=False):
        """Save model along with the training config"""
        if output_dir is None:
            output_dir = self.args.output_dir

        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        yaml.dump(self.training_config, open(
            os.path.join(output_dir, "training_config.yml"), "w", encoding="utf-8"))
