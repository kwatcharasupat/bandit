from typing import Dict, List, Optional

import torch

from core.model._spectral import _SpectralComponent


class BaseIdealMask(
        _SpectralComponent
):
    def __init__(
            self,
            stems: List[str],
            fs: int = 44100,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
    ) -> None:
        super().__init__(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

        self.stems = stems
        self.fs = fs
        self.bypass_fader = True

    def compute_mask(self, stem, mixture):
        raise NotImplementedError
    def forward(self, batch):
        audio = batch["audio"]

        with torch.no_grad():
            batch["spectrogram"] = {stem: self.stft(audio[stem]) for stem in
                                    audio}

        X = batch["spectrogram"]["mixture"]
        length = batch["audio"]["mixture"].shape[-1]

        output = {"spectrogram": {}, "audio": {}}

        for stem, spec in batch["spectrogram"].items():
            if stem == "mixture":
                continue

            # print("stem", stem)

            mask = self.compute_mask(spec, X)

            output["spectrogram"][stem] = mask * X
            output["audio"][stem] = self.istft(output["spectrogram"][stem], length)

        return batch, output


class IdentityMask(BaseIdealMask):
    def __init__(self,
            stems: List[str],
            fs: int = 44100,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
    ) -> None:
        super().__init__(
                stems=stems,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )


    def compute_mask(self, stem, mixture):
        return torch.ones_like(mixture)



class IdealAmplitudeMask(BaseIdealMask):
    def __init__(
            self,
            stems: List[str],
            fs: int = 44100,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
    ) -> None:
        super().__init__(
                stems=stems,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

    def compute_mask(self, stem, mixture):
        astem = torch.abs(stem)
        amix = torch.abs(mixture)
        return torch.where(
                (astem < 1e-8) & (amix < 1e-8),
                torch.ones_like(astem),
                torch.where(
                        amix < 1e-8,
                        torch.ones_like(astem),
                        astem / amix
                )
        )


class PhaseSensitiveFilter(BaseIdealMask):
    def __init__(
            self,
            stems: List[str],
            fs: int = 44100,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
    ) -> None:
        super().__init__(
                stems=stems,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

    def compute_mask(self, stem, mixture):
        astem = torch.abs(stem)
        amix = torch.abs(mixture)

        angle = torch.angle(stem) - torch.angle(mixture)

        return torch.where(
                (astem < 1e-8) & (amix < 1e-8),
                torch.ones_like(astem),
                torch.where(
                        amix < 1e-8,
                        torch.ones_like(astem),
                        astem * torch.cos(angle) / amix
                )
        )




class IdealWienerMask(BaseIdealMask):
    def __init__(
            self,
            stems: List[str],
            fs: int = 44100,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
    ) -> None:
        super().__init__(
                stems=stems,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

    def compute_mask(self, stem, mixture):
        astem = torch.abs(stem)
        amix = torch.abs(mixture)
        noise = mixture - stem
        anoise = torch.abs(noise)
        return torch.where(
                (astem < 1e-8) & (amix < 1e-8),
                torch.ones_like(astem),
                torch.where(
                        amix < 1e-8,
                        torch.ones_like(astem),
                        torch.square(astem) / (torch.square(astem) + torch.square(anoise))
                )
        )



class IdealRatioMask(BaseIdealMask):
    def __init__(
            self,
            stems: List[str],
            fs: int = 44100,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
    ) -> None:
        super().__init__(
                stems=stems,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

    def compute_mask(self, stem, mixture):
        astem = torch.abs(stem)
        amix = torch.abs(mixture)
        noise = mixture - stem
        anoise = torch.abs(noise)
        return torch.where(
                (astem < 1e-8) & (amix < 1e-8),
                torch.ones_like(astem),
                torch.where(
                        amix < 1e-8,
                        torch.ones_like(astem),
                        astem / (astem + anoise)
                )
        )



class IdealBinaryMask(BaseIdealMask):
    def __init__(
            self,
            stems: List[str],
            fs: int = 44100,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
    ) -> None:
        super().__init__(
                stems=stems,
                fs=fs,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_fn=window_fn,
                wkwargs=wkwargs,
                power=power,
                center=center,
                normalized=normalized,
                pad_mode=pad_mode,
                onesided=onesided,
        )

    def compute_mask(self, stem, mixture):
        astem = torch.abs(stem)
        # amix = torch.abs(mixture)
        noise = mixture - stem
        anoise = torch.abs(noise)
        return torch.where(
                astem > anoise,
                torch.ones_like(astem),
                torch.zeros_like(astem)
        )