import yaml
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np
import librosa
from scipy.io.wavfile import write
from utils import ignore_warnings; ignore_warnings()
from utils import parse_yaml, load_ss_model
from models.clap_encoder import CLAP_Encoder


def build_audiosep(config_yaml, checkpoint_path, device):
    configs = parse_yaml(config_yaml)
    
    query_encoder = CLAP_Encoder().eval()
    model = load_ss_model(
            configs=configs,
            checkpoint_path=checkpoint_path,
            query_encoder=query_encoder
        ).eval().to(device)

    print(f'Load AudioSep model from [{checkpoint_path}]')
    return model


def inference(model, audio_file, text, output_file, device='cuda'):
    print(f'Separate audio from [{audio_file}] with textual query [{text}]')
    mixture, fs = librosa.load(audio_file, sr=32000, mono=True)
    with torch.no_grad():
        text = [text]

        conditions = model.query_encoder.get_query_embed(
            modality='text',
            text=text,
            device=device
        )

        input_dict = {
            "mixture": torch.Tensor(mixture)[None, None, :].to(device),
            "condition": conditions,
        } 

        sep_segment = model.ss_model(input_dict)["waveform"]

        sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()

        write(output_file, 32000, np.round(sep_segment * 32767).astype(np.int16))
        print(f'Write separated audio to [{output_file}]')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_audiosep(
        config_yaml='config/audiosep_base.yaml', 
        checkpoint_path='checkpoint/step=3920000.ckpt', 
        device=device)

    audio_file = '/mnt/bn/data-xubo/project/AudioShop/YT_audios/Y3VHpLxtd498.wav'
    text = 'pigeons are cooing in the background'
    output_file='separated_audio.wav'
    
    inference(model, audio_file, text, output_file, device)

    




