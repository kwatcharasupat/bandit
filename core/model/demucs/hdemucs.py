from typing import Dict, List

import torch
from torch import nn
import torchaudio as ta


class HDemucsWrapper(nn.Module):
    def __init__(
            self,
            stems: List[str],
            fs: int = 44100,
            audio_channels: int = 1,
            channels: int = 48,
            growth: int = 2,
            nfft: int = 4096,
            depth: int = 6,
            freq_emb: float = 0.2,
            emb_scale: int = 10,
            emb_smooth: bool = True,
            kernel_size: int = 8,
            time_stride: int = 2,
            stride: int = 4,
            context: int = 1,
            context_enc: int = 0,
            norm_starts: int = 4,
            norm_groups: int = 4,
            dconv_depth: int = 2,
            dconv_comp: int = 4,
            dconv_attn: int = 4,
            dconv_lstm: int = 4,
            dconv_init: float = 0.0001
            ):
        super().__init__(        )

        self.demucs = ta.models.HDemucs(
                sources=stems,
                audio_channels=audio_channels,
                channels=channels,
                growth=growth,
                nfft=nfft,
                depth=depth,
                freq_emb=freq_emb,
                emb_scale=emb_scale,
                emb_smooth=emb_smooth,
                kernel_size=kernel_size,
                time_stride=time_stride,
                stride=stride,
                context=context,
                context_enc=context_enc,
                norm_starts=norm_starts,
                norm_groups=norm_groups,
                dconv_depth=dconv_depth,
                dconv_comp=dconv_comp,
                dconv_attn=dconv_attn,
                dconv_lstm=dconv_lstm,
                dconv_init=dconv_init
        )

        self.stems = stems
        self.fs = fs

    def forward(self, batch):

        x = batch['audio']["mixture"]
        # print(x.shape)

        shat = self.demucs(x) # (batch_size, num_sources, channel, num_frames)
        # print(output.shape)

        output = {
                "audio": {
                    stem: shat[:, i, :, :] for i, stem in enumerate(self.stems)
                }
        }

        return batch, output
