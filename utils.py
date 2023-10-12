import os
import datetime
import json
import logging
import librosa
import pickle
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import yaml
from models.audiosep import AudioSep, get_model_class


def ignore_warnings():
    import warnings
    # Ignore UserWarning from torch.meshgrid
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.functional')

    # Refined regex pattern to capture variations in the warning message
    pattern = r"Some weights of the model checkpoint at roberta-base were not used when initializing RobertaModel: \['lm_head\..*'\].*"
    warnings.filterwarnings('ignore', message=pattern)



def create_logging(log_dir, filemode):
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging


def float32_to_int16(x: float) -> int:
    x = np.clip(x, a_min=-1, a_max=1)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: int) -> float:
    return (x / 32767.0).astype(np.float32)


def parse_yaml(config_yaml: str) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


def get_audioset632_id_to_lb(ontology_path: str) -> Dict:
    r"""Get AudioSet 632 classes ID to label mapping."""
    
    audioset632_id_to_lb = {}

    with open(ontology_path) as f:
        data_list = json.load(f)

    for e in data_list:
        audioset632_id_to_lb[e["id"]] = e["name"]

    return audioset632_id_to_lb


def load_pretrained_panns(
    model_type: str,
    checkpoint_path: str,
    freeze: bool
) -> nn.Module:
    r"""Load pretrained pretrained audio neural networks (PANNs).

    Args:
        model_type: str, e.g., "Cnn14"
        checkpoint_path, str, e.g., "Cnn14_mAP=0.431.pth"
        freeze: bool

    Returns:
        model: nn.Module
    """

    if model_type == "Cnn14":
        Model = Cnn14

    elif model_type == "Cnn14_DecisionLevelMax":
        Model = Cnn14_DecisionLevelMax

    else:
        raise NotImplementedError

    model = Model(sample_rate=32000, window_size=1024, hop_size=320,
                  mel_bins=64, fmin=50, fmax=14000, classes_num=527)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


def energy(x):
    return torch.mean(x ** 2)


def magnitude_to_db(x):
    eps = 1e-10
    return 20. * np.log10(max(x, eps))


def db_to_magnitude(x):
    return 10. ** (x / 20)


def ids_to_hots(ids, classes_num, device):
    hots = torch.zeros(classes_num).to(device)
    for id in ids:
        hots[id] = 1
    return hots


def calculate_sdr(
    ref: np.ndarray,
    est: np.ndarray,
    eps=1e-10
) -> float:
    r"""Calculate SDR between reference and estimation.

    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    reference = ref
    noise = est - reference


    numerator = np.clip(a=np.mean(reference ** 2), a_min=eps, a_max=None)

    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)

    sdr = 10. * np.log10(numerator / denominator)

    return sdr


def calculate_sisdr(ref, est):
    r"""Calculate SDR between reference and estimation.

    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """

    eps = np.finfo(ref.dtype).eps

    reference = ref.copy()
    estimate = est.copy()
    
    reference = reference.reshape(reference.size, 1)
    estimate = estimate.reshape(estimate.size, 1)

    Rss = np.dot(reference.T, reference)
    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

    e_true = a * reference
    e_res = estimate - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    sisdr = 10 * np.log10((eps+ Sss)/(eps + Snn))

    return sisdr 


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = "{}_{}.pkl".format(
            os.path.splitext(self.statistics_path)[0],
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

        self.statistics_dict = {"balanced_train": [], "test": []}

    def append(self, steps, statistics, split, flush=True):
        statistics["steps"] = steps
        self.statistics_dict[split].append(statistics)

        if flush:
            self.flush()

    def flush(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, "wb"))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, "wb"))
        logging.info("    Dump statistics to {}".format(self.statistics_path))
        logging.info("    Dump statistics to {}".format(self.backup_statistics_path))


def get_mean_sdr_from_dict(sdris_dict):
    mean_sdr = np.nanmean(list(sdris_dict.values()))
    return mean_sdr


def remove_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    r"""Remove silent frames."""
    window_size = int(sample_rate * 0.1)
    threshold = 0.02

    frames = librosa.util.frame(x=audio, frame_length=window_size, hop_length=window_size).T
    # shape: (frames_num, window_size)

    new_frames = get_active_frames(frames, threshold)
    # shape: (new_frames_num, window_size)

    new_audio = new_frames.flatten()
    # shape: (new_audio_samples,)

    return new_audio


def get_active_frames(frames: np.ndarray, threshold: float) -> np.ndarray:
    r"""Get active frames."""

    energy = np.max(np.abs(frames), axis=-1)
    # shape: (frames_num,)

    active_indexes = np.where(energy > threshold)[0]
    # shape: (new_frames_num,)

    new_frames = frames[active_indexes]
    # shape: (new_frames_num,)

    return new_frames


def repeat_to_length(audio: np.ndarray, segment_samples: int) -> np.ndarray:
    r"""Repeat audio to length."""
    
    repeats_num = (segment_samples // audio.shape[-1]) + 1
    audio = np.tile(audio, repeats_num)[0 : segment_samples]

    return audio

def calculate_segmentwise_sdr(ref, est, hop_samples, return_sdr_list=False):
    min_len = min(ref.shape[-1], est.shape[-1])
    pointer = 0
    sdrs = []
    while pointer + hop_samples < min_len:
        sdr = calculate_sdr(
            ref=ref[:, pointer : pointer + hop_samples], 
            est=est[:, pointer : pointer + hop_samples],
        )
        sdrs.append(sdr)
        pointer += hop_samples

    sdr = np.nanmedian(sdrs)

    if return_sdr_list:
        return sdr, sdrs
    else:
        return sdr


def loudness(data, input_loudness, target_loudness):
    """ Loudness normalize a signal.
    
    Normalize an input signal to a user loudness in dB LKFS.   

    Params
    -------
    data : torch.Tensor
        Input multichannel audio data.
    input_loudness : float
        Loudness of the input in dB LUFS. 
    target_loudness : float
        Target loudness of the output in dB LUFS.
        
    Returns
    -------
    output : torch.Tensor
        Loudness normalized output data.
    """    
        
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = target_loudness - input_loudness
    gain = torch.pow(10.0, delta_loudness / 20.0)

    output = gain * data

    # check for potentially clipped samples
    # if torch.max(torch.abs(output)) >= 1.0:
    #     warnings.warn("Possible clipped samples in output.")

    return output


def load_ss_model(
    configs: Dict,
    checkpoint_path: str,
    query_encoder: nn.Module
) -> nn.Module:
    r"""Load trained universal source separation model.

    Args:
        configs (Dict)
        checkpoint_path (str): path of the checkpoint to load
        device (str): e.g., "cpu" | "cuda"

    Returns:
        pl_model: pl.LightningModule
    """

    ss_model_type = configs["model"]["model_type"]
    input_channels = configs["model"]["input_channels"]
    output_channels = configs["model"]["output_channels"]
    condition_size = configs["model"]["condition_size"]
    
    # Initialize separation model
    SsModel = get_model_class(model_type=ss_model_type)

    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    # Load PyTorch Lightning model
    pl_model = AudioSep.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        ss_model=ss_model,
        waveform_mixer=None,
        query_encoder=query_encoder,
        loss_function=None,
        optimizer_type=None,
        learning_rate=None,
        lr_lambda_func=None,
        map_location=torch.device('cpu'),
    )

    return pl_model


def parse_yaml(config_yaml: str) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)