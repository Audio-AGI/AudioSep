import random
import sre_compile
import numpy as np
import torch
import torch.nn as nn
import pyloudnorm as pyln


class SegmentMixer(nn.Module):
    def __init__(self, max_mix_num, lower_db, higher_db):
        super(SegmentMixer, self).__init__()

        self.max_mix_num = max_mix_num
        self.loudness_param = {
            'lower_db': lower_db,
            'higher_db': higher_db,
        }

    def __call__(self, waveforms):
        
        batch_size = waveforms.shape[0]

        data_dict = {
            'segment': [],
            'mixture': [],
        }

        for n in range(0, batch_size):

            segment = waveforms[n].clone()

            # create zero tensors as the background template
            noise = torch.zeros_like(segment)

            mix_num = random.randint(2, self.max_mix_num)
            assert mix_num >= 2

            for i in range(1, mix_num):
                next_segment = waveforms[(n + i) % batch_size]
                rescaled_next_segment = dynamic_loudnorm(audio=next_segment, reference=segment, **self.loudness_param)
                noise += rescaled_next_segment

            # randomly normalize background noise
            noise = dynamic_loudnorm(audio=noise, reference=segment, **self.loudness_param)

            # create audio mixyure
            mixture = segment + noise

            # declipping if need be
            max_value = torch.max(torch.abs(mixture))
            if max_value > 1:
                segment *= 0.9 / max_value
                mixture *= 0.9 / max_value

            data_dict['segment'].append(segment)
            data_dict['mixture'].append(mixture)

        for key in data_dict.keys():
            data_dict[key] = torch.stack(data_dict[key], dim=0)

        # return data_dict
        return data_dict['mixture'], data_dict['segment']


def rescale_to_match_energy(segment1, segment2):

    ratio = get_energy_ratio(segment1, segment2)
    rescaled_segment1 = segment1 / ratio
    return rescaled_segment1 


def get_energy(x):
    return torch.mean(x ** 2)


def get_energy_ratio(segment1, segment2):

    energy1 = get_energy(segment1)
    energy2 = max(get_energy(segment2), 1e-10)
    ratio = (energy1 / energy2) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50)
    return ratio


def dynamic_loudnorm(audio, reference, lower_db=-10, higher_db=10): 
    rescaled_audio = rescale_to_match_energy(audio, reference)
    
    delta_loudness = random.randint(lower_db, higher_db)

    gain = np.power(10.0, delta_loudness / 20.0)

    return gain * rescaled_audio


def torch_to_numpy(tensor):
    """Convert a PyTorch tensor to a NumPy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise ValueError("Input must be a PyTorch tensor.")


def numpy_to_torch(array):
    """Convert a NumPy array to a PyTorch tensor."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise ValueError("Input must be a NumPy array.")


# decayed
def random_loudness_norm(audio, lower_db=-35, higher_db=-15, sr=32000):
    device = audio.device
    audio = torch_to_numpy(audio.squeeze(0))
    # randomly select a norm volume
    norm_vol = random.randint(lower_db, higher_db)

    # measure the loudness first 
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    # loudness normalize audio
    normalized_audio = pyln.normalize.loudness(audio, loudness, norm_vol)

    normalized_audio = numpy_to_torch(normalized_audio).unsqueeze(0)
    
    return normalized_audio.to(device)
    
