import os
import sys
import re
from typing import Dict, List

import csv
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


class VGGSoundEvaluator:
    def __init__(
        self,
        sampling_rate=32000
    ) -> None:
        r"""VGGSound evaluator.

        Args:
            data_recipe (str): dataset split, 'yan' 
        Returns:
            None
        """

        self.sampling_rate = sampling_rate

        with open('evaluation/metadata/vggsound_eval.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        
        self.eval_list = eval_list
        self.audio_dir = 'evaluation/data/vggsound'

    def __call__(
        self,
        pl_model: pl.LightningModule
    ) -> Dict:
        r"""Evalute."""

        print(f'Evaluation on VGGSound+ with [text label] queries.')
        
        pl_model.eval()
        device = pl_model.device

        sisdrs_list = []
        sdris_list = []
        sisdris_list = []
        

        with torch.no_grad():
            for eval_data in tqdm(self.eval_list):

                # labels, source_path, mixture_path = eval_data
                file_id, mix_wav, s0_wav, s0_text, s1_wav, s1_text = eval_data

                labels = s0_text

                mixture_path = os.path.join(self.audio_dir, mix_wav)
                source_path = os.path.join(self.audio_dir, s0_wav)


                source, fs = librosa.load(source_path, sr=self.sampling_rate, mono=True)
                mixture, fs = librosa.load(mixture_path, sr=self.sampling_rate, mono=True)

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)
                                
                text = [labels]
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

                sisdr_no_sep = calculate_sisdr(ref=source, est=mixture)
                sisdr = calculate_sisdr(ref=source, est=sep_segment)
                sisdri = sisdr - sisdr_no_sep

                sisdrs_list.append(sisdr)
                sdris_list.append(sdri)
                sisdris_list.append(sisdri)


        mean_sisdr = np.mean(sisdrs_list)
        mean_sdri = np.mean(sdris_list)

        return mean_sisdr, mean_sdri