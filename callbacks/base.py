import os
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only


class CheckpointEveryNSteps(pl.Callback):
    def __init__(
        self,
        checkpoints_dir,
        save_step_frequency,
    ) -> None:
        r"""Save a checkpoint every N steps.

        Args:
            checkpoints_dir (str): directory to save checkpoints
            save_step_frequency (int): save checkpoint every N step
        """

        self.checkpoints_dir = checkpoints_dir
        self.save_step_frequency = save_step_frequency

    @rank_zero_only
    def on_train_batch_end(self, *args, **kwargs) -> None:
        r"""Save a checkpoint every N steps."""

        trainer = args[0]
        global_step = trainer.global_step

        if global_step == 1 or global_step % self.save_step_frequency == 0:

            ckpt_path = os.path.join(
                self.checkpoints_dir,
                "step={}.ckpt".format(global_step))
            trainer.save_checkpoint(ckpt_path)
            print("Save checkpoint to {}".format(ckpt_path))
