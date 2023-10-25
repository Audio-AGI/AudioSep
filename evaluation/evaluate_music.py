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


class MUSICEvaluator:
    def __init__(
        self,
        sampling_rate=32000
    ) -> None:

        self.sampling_rate = sampling_rate

        with open('evaluation/metadata/music_eval.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        
        self.eval_list = eval_list
        self.audio_dir = 'evaluation/data/music'

        self.source_types = [
        "acoustic guitar", 
        "violin", 
        "accordion", 
        "xylophone", 
        "erhu", 
        "trumpet", 
        "tuba", 
        "cello", 
        "flute", 
        "saxophone"]

    def __call__(
        self,
        pl_model: pl.LightningModule
    ) -> Dict:
        r"""Evaluate."""

        print(f'Evaluation on MUSIC Test with [text label] queries.')
        
        pl_model.eval()
        device = pl_model.device

        sisdrs_list = {source_type: [] for source_type in self.source_types}
        sdris_list = {source_type: [] for source_type in self.source_types}

        with torch.no_grad():
            for eval_data in tqdm(self.eval_list):

                idx, caption, _, _, = eval_data

                source_path = os.path.join(self.audio_dir, f'segment-{idx}.wav')
                mixture_path = os.path.join(self.audio_dir, f'mixture-{idx}.wav')

                source, fs = librosa.load(source_path, sr=self.sampling_rate, mono=True)
                mixture, fs = librosa.load(mixture_path, sr=self.sampling_rate, mono=True)

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)
                                
                text = [caption]

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

                sisdrs_list[caption].append(sisdr)
                sdris_list[caption].append(sdri)

        mean_sisdr_list = []
        mean_sdri_list = []
        
        for source_class in self.source_types:
            sisdr = np.mean(sisdrs_list[source_class])
            sdri = np.mean(sdris_list[source_class])
            mean_sisdr_list.append(sisdr)
            mean_sdri_list.append(sdri)
        
        mean_sdri = np.mean(mean_sdri_list)
        mean_sisdr = np.mean(mean_sisdr_list)
        
        return mean_sisdr, mean_sdri
