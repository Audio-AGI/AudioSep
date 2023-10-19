import os
import sys
import re
from typing import Dict, List

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pathlib
import librosa
import lightning.pytorch as pl
from models.clap_encoder import CLAP_Encoder

sys.path.append('../AudioSep/')
from utils import (
    load_ss_model,
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)


meta_csv_file = "evaluation/metadata/class_labels_indices.csv"
df = pd.read_csv(meta_csv_file, sep=',')

IDS = df['mid'].tolist()
LABELS = df['display_name'].tolist()

CLASSES_NUM = len(LABELS)

IX_TO_LB = {i : label for i, label in enumerate(LABELS)}


class AudioSetEvaluator:
    def __init__(
        self,
        audios_dir='evaluation/data/audioset',
        classes_num=527,
        sampling_rate=32000,
        number_per_class=10,
    ) -> None:
        r"""AudioSet evaluator.

        Args:
            audios_dir (str): directory of evaluation segments
            classes_num (int): the number of sound classes
            number_per_class (int), the number of samples to evaluate for each sound class

        Returns:
            None
        """

        self.audios_dir = audios_dir
        self.classes_num = classes_num
        self.number_per_class = number_per_class
        self.sampling_rate = sampling_rate

    @torch.no_grad()
    def __call__(
        self,
        pl_model: pl.LightningModule
    ) -> Dict:
        r"""Evalute."""

        pl_model.eval()

        sisdrs_dict = {class_id: [] for class_id in range(self.classes_num)}
        sdris_dict = {class_id: [] for class_id in range(self.classes_num)}
        
        print('Evaluation on AudioSet with [text label] queries.')
        
        for class_id in tqdm(range(self.classes_num)):

            sub_dir = os.path.join(
                self.audios_dir,
                "class_id={}".format(class_id))

            audio_names = self._get_audio_names(audios_dir=sub_dir)

            for audio_index, audio_name in enumerate(audio_names):

                if audio_index == self.number_per_class:
                    break

                source_path = os.path.join(
                    sub_dir, "{},source.wav".format(audio_name))
                mixture_path = os.path.join(
                    sub_dir, "{},mixture.wav".format(audio_name))

                source, fs = librosa.load(source_path, sr=self.sampling_rate, mono=True)
                mixture, fs = librosa.load(mixture_path, sr=self.sampling_rate, mono=True)

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)

                device = pl_model.device

                text = [IX_TO_LB[class_id]]

                conditions = pl_model.query_encoder.get_query_embed(
                    modality='text',
                    text=text,
                    device=device 
                )

                input_dict = {
                    "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                    "condition": conditions,
                }

                sep_segment = pl_model.ss_model(input_dict)["waveform"]
                # sep_segment: (batch_size=1, channels_num=1, segment_samples)

                sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
                # sep_segment: (segment_samples,)

                sdr = calculate_sdr(ref=source, est=sep_segment)
                sdri = sdr - sdr_no_sep
                sisdr = calculate_sisdr(ref=source, est=sep_segment)


                sisdrs_dict[class_id].append(sisdr)
                sdris_dict[class_id].append(sdri)


        stats_dict = {
            "sisdrs_dict": sisdrs_dict,
            "sdris_dict": sdris_dict,
        }

        return stats_dict

    def _get_audio_names(self, audios_dir: str) -> List[str]:
        r"""Get evaluation audio names."""
        audio_names = sorted(os.listdir(audios_dir))

        audio_names = [audio_name for audio_name in audio_names if '.wav' in audio_name]
        
        audio_names = [
            re.search(
                "(.*),(mixture|source).wav",
                audio_name).group(1) for audio_name in audio_names]

        audio_names = sorted(list(set(audio_names)))

        return audio_names

    @staticmethod
    def get_median_metrics(stats_dict, metric_type):
        class_ids = stats_dict[metric_type].keys()
        median_stats_dict = {
            class_id: np.nanmedian(
                stats_dict[metric_type][class_id]) for class_id in class_ids}
        return median_stats_dict
