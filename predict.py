# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from cog import BasePredictor, Input, Path

from pipeline import build_audiosep, inference


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.model = build_audiosep(
            config_yaml="config/audiosep_base.yaml",
            checkpoint_path="checkpoint/audiosep_base_4M_steps.ckpt",
            device="cuda",
        )

    def predict(
        self,
        audio_file: Path = Input(description="Input audio file."),
        text: str = Input(description="Input text.", default="water drops"),
    ) -> Path:
        """Run a single prediction on the model"""

        output_file = "/tmp/separated_audio.wav"

        # AudioSep processes the audio at 32 kHz sampling rate
        inference(self.model, str(audio_file), text, output_file, "cuda")
        return Path(output_file)
