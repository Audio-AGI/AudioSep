import random
import torch
import torch.nn as nn
import torchaudio
from models.CLAP.open_clip import create_model
from models.CLAP.training.data import get_audio_features
from transformers import RobertaTokenizer
from utils import ignore_warnings; ignore_warnings()


class CLAP_Encoder(nn.Module):
    def __init__(
        self,
        pretrained_path='checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt',
        sampling_rate=32000,
        amodel = "HTSAT-base",
    ):
        super().__init__()
        self.device = "cpu"
        self.precision = "fp32"
        self.amodel = amodel  # or 'PANN-14'
        self.tmodel = "roberta"  # the best text encoder in our training
        self.enable_fusion = False  # False if you do not want to use the fusion model
        self.fusion_type = "aff_2d"
        self.pretrained = pretrained_path
        self.sampling_rate = sampling_rate
        self.tokenize = RobertaTokenizer.from_pretrained("roberta-base")
        
        self.model, self.model_cfg = create_model(
            self.amodel,
            self.tmodel,
            self.pretrained,
            precision=self.precision,
            device=self.device,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        self.encoder_type = 'CLAP'

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret

    def _get_audio_embed(self, batch):
        # batch: [B, samples]
        with torch.no_grad():
            audio_dict_list = []
            assert (
                self.sampling_rate == 32000
            ), "We only support 32000 sampling rate"
                
            # batch: [bs, 1, t-samples]
            batch = torchaudio.functional.resample(
                batch, orig_freq=self.sampling_rate, new_freq=48000
            )
            for waveform in self.batch_to_list(batch):
                audio_dict = {}
                audio_dict = get_audio_features(
                    audio_dict,
                    waveform,
                    480000,
                    data_truncating="fusion",
                    data_filling="repeatpad",
                    audio_cfg=self.model_cfg["audio_cfg"],
                )
                audio_dict_list.append(audio_dict)
                # [bs, 512]
                embed = self.model.get_audio_embedding(audio_dict_list)
            
                return embed.detach()

    def _get_text_embed(self, batch):
        double_batch = False
        if len(batch) == 1:
            batch = batch * 2
            double_batch = True
        with torch.no_grad():
            # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
            text_data = self.tokenizer(batch)
            embed = self.model.get_text_embedding(text_data)
        if double_batch:
            embed = embed[0].unsqueeze(0)
        
        return embed.detach()


    def get_query_embed(self, modality, audio=None, text=None, use_text_ratio=0.5, device=None):
        if modality == 'audio':
            embed = self._get_audio_embed(audio)
        elif modality == 'text':
            embed = self._get_text_embed(text)
        elif modality == 'hybird':
            if random.random() > use_text_ratio:
                embed = self._get_audio_embed(audio)
            else:
                embed = self._get_text_embed(text)
        else:
            raise NotImplementedError("Please check flag 'training_modality'.")

        return embed.float()

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}
