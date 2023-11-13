import warnings
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn.modules import activation

from core.model.bsrnn.utils import (
    band_widths_from_specs,
    check_no_gap,
    check_no_overlap,
    check_nonzero_bandwidth,
)


class BaseNormMLP(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            mlp_dim: int,
            bandwidth: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs=None,
            complex_mask: bool = True, ):

        super().__init__()
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}
        self.hidden_activation_kwargs = hidden_activation_kwargs
        self.norm = nn.LayerNorm(emb_dim)
        self.hidden = torch.jit.script(nn.Sequential(
                nn.Linear(in_features=emb_dim, out_features=mlp_dim),
                activation.__dict__[hidden_activation](
                        **self.hidden_activation_kwargs
                ),
        ))

        self.bandwidth = bandwidth
        self.in_channel = in_channel

        self.complex_mask = complex_mask
        self.reim = 2 if complex_mask else 1
        self.glu_mult = 2


class NormMLP(BaseNormMLP):
    def __init__(
            self,
            emb_dim: int,
            mlp_dim: int,
            bandwidth: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs=None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__(
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                bandwidth=bandwidth,
                in_channel=in_channel,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
        )

        self.output = torch.jit.script(
                nn.Sequential(
                        nn.Linear(
                                in_features=mlp_dim,
                                out_features=bandwidth * in_channel * self.reim * 2,
                        ),
                        nn.GLU(dim=-1),
                )
        )

    def reshape_output(self, mb):
        # print(mb.shape)
        batch, n_time, _ = mb.shape
        if self.complex_mask:
            mb = mb.reshape(
                    batch,
                    n_time,
                    self.in_channel,
                    self.bandwidth,
                    self.reim
            ).contiguous()
            # print(mb.shape)
            mb = torch.view_as_complex(
                    mb
            )  # (batch, n_time, in_channel, bandwidth)
        else:
            mb = mb.reshape(batch, n_time, self.in_channel, self.bandwidth)

        mb = torch.permute(
                mb,
                (0, 2, 3, 1)
        )  # (batch, in_channel, bandwidth, n_time)

        return mb

    def forward(self, qb):
        # qb = (batch, n_time, emb_dim)

        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("qb0")


        qb = self.norm(qb)  # (batch, n_time, emb_dim)

        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("qb1")

        qb = self.hidden(qb)  # (batch, n_time, mlp_dim)
        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("qb2")
        mb = self.output(qb)  # (batch, n_time, bandwidth * in_channel * reim)
        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("mb")
        mb = self.reshape_output(mb)  # (batch, in_channel, bandwidth, n_time)

        return mb


class MultAddNormMLP(NormMLP):
    def __init__(self, emb_dim: int, mlp_dim: int, bandwidth: int, in_channel: int | None, hidden_activation: str = "Tanh", hidden_activation_kwargs=None, complex_mask: bool = True) -> None:
        super().__init__(emb_dim, mlp_dim, bandwidth, in_channel, hidden_activation, hidden_activation_kwargs, complex_mask)

        self.output2 = torch.jit.script(
                nn.Sequential(
                        nn.Linear(
                                in_features=mlp_dim,
                                out_features=bandwidth * in_channel * self.reim * 2,
                        ),
                        nn.GLU(dim=-1),
                )
        )

    def forward(self, qb):

        qb = self.norm(qb)  # (batch, n_time, emb_dim)
        qb = self.hidden(qb)  # (batch, n_time, mlp_dim)
        mmb = self.output(qb)  # (batch, n_time, bandwidth * in_channel * reim)
        mmb = self.reshape_output(mmb)  # (batch, in_channel, bandwidth, n_time)
        amb = self.output2(qb)  # (batch, n_time, bandwidth * in_channel * reim)
        amb = self.reshape_output(amb)  # (batch, in_channel, bandwidth, n_time)

        return mmb, amb


class BaseKernelNormMLP(BaseNormMLP):
    def __init__(
            self,
            bandwidth: int,
            in_channel: Optional[int],
            emb_dim: int,
            mlp_dim: int,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs=None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__(
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                bandwidth=bandwidth,
                in_channel=in_channel,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
        )

    def reshape_output(self, mb):

        mbshape = mb.shape
        batch = mbshape[0]
        n_time = mbshape[-1]
        # (batch, in_channel * reim * kernel_freq * kernel_time, bandwidth, n_time)

        if self.complex_mask:
            mb = mb.reshape(
                    batch,
                    self.in_channel,
                    self.mask_kernel_freq,
                    self.mask_kernel_time,
                    2,
                    self.bandwidth,
                    n_time
            )  # (batch, in_channel, reim, kernel_freq, kernel_time, bandwidth, n_time)

            mb = torch.permute(
                    mb,
                    (0, 1, 2, 3, 5, 6, 4)
            )  # (batch, in_channel, kernel_freq, kernel_time, bandwidth, n_time)
            mb = torch.view_as_complex(
                    mb.contiguous()
            )  # (batch, in_channel, kernel_freq, kernel_time, bandwidth, n_time)
        else:
            mb = mb.reshape(
                    batch,
                    self.in_channel,
                    self.mask_kernel_freq,
                    self.mask_kernel_time,
                    self.bandwidth,
                    n_time
            )  # (batch, in_channel, kernel_freq, kernel_time, bandwidth, n_time)

        return mb


class KernelNormMLP(BaseKernelNormMLP):
    def __init__(
            self,
            emb_dim: int,
            mlp_dim: int,
            bandwidth: int,
            mask_kernel_freq: int,
            mask_kernel_time: int,
            conv_kernel_freq: int,
            conv_kernel_time: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs=None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__(
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                bandwidth=bandwidth,
                in_channel=in_channel,
        )

        self.output = nn.Linear(
                in_features=mlp_dim,
                out_features=bandwidth * in_channel * self.reim * self.glu_mult,
        )

        self.mask_kernel_freq = mask_kernel_freq
        self.mask_kernel_time = mask_kernel_time

        self.conv = nn.Sequential(
                nn.Conv2d(
                        in_channels=in_channel * self.reim * self.glu_mult,
                        out_channels=in_channel * self.reim * mask_kernel_freq * mask_kernel_time * self.glu_mult,
                        kernel_size=(conv_kernel_freq, conv_kernel_time),
                        padding="same",
                        bias=False,
                ),
                nn.GLU(dim=1)
        )

    def forward(self, qb):
        # qb = (batch, n_time, emb_dim)

        qb = self.norm(qb)  # (batch, n_time, emb_dim)
        qb = self.hidden(qb)  # (batch, n_time, mlp_dim)
        mb = self.output(qb)
        # (batch, n_time, bandwidth * in_channel * reim * glu_mult)
        batch, n_time, _ = mb.shape
        mb = mb.reshape(
                batch,
                n_time,
                self.bandwidth,
                self.in_channel * self.reim * self.glu_mult
        )
        mb = torch.permute(mb, (0, 3, 2, 1))
        # (batch, in_channel * reim, bandwidth, n_time)
        mb = self.conv(mb)
        mb = self.reshape_output(mb)

        return mb


class KernelNormMLP2(BaseKernelNormMLP):
    def __init__(
            self,
            emb_dim: int,
            mlp_dim: int,
            bandwidth: int,
            mask_kernel_freq: int,
            mask_kernel_time: int,
            conv_kernel_freq: int,
            conv_kernel_time: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs=None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__(
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                bandwidth=bandwidth,
                in_channel=in_channel,
        )

        self.mask_kernel_freq = mask_kernel_freq
        self.mask_kernel_time = mask_kernel_time

        self.output = nn.Sequential(
                nn.Conv1d(
                        in_channels=mlp_dim,
                        out_channels=bandwidth * in_channel * self.reim * self.glu_mult * mask_kernel_freq * mask_kernel_time,
                        kernel_size=conv_kernel_time,
                        padding="same"
                ),
                nn.GLU(dim=1),
        )

        if conv_kernel_freq is not None:
            warnings.warn(
                    "`conv_kernel_freq` is not None, but it is not used in `KernelNormMLP2`. "
            )

        # torch.nn.init.dirac_(self.conv.weight)

    def forward(self, qb):
        # qb = (batch, n_time, emb_dim)

        qb = self.norm(qb)  # (batch, n_time, emb_dim)

        qb = self.hidden(qb)  # (batch, n_time, mlp_dim)
        qb = torch.permute(qb, (0, 2, 1))  # (batch, mlp_dim, n_time)
        mb = self.output(qb)
        # (batch, bandwidth * in_channel * reim * kernel_freq * kernel_time, n_time)

        mb = self.reshape_output(mb)

        return mb


class KernelNormMLP3(BaseKernelNormMLP):
    def __init__(
            self,
            emb_dim: int,
            mlp_dim: int,
            bandwidth: int,
            mask_kernel_freq: int,
            mask_kernel_time: int,
            conv_kernel_freq: int,
            conv_kernel_time: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs=None,
            complex_mask: bool = True,
    ) -> None:
        super().__init__(
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                bandwidth=bandwidth,
                in_channel=in_channel,
        )

        self.mask_kernel_freq = mask_kernel_freq
        self.mask_kernel_time = mask_kernel_time

        self.hidden_conv = nn.Sequential(
                nn.Conv1d(
                        in_channels=mlp_dim,
                        out_channels=mlp_dim * mask_kernel_freq * mask_kernel_time,
                        kernel_size=conv_kernel_time,
                        padding="same",
                        groups=mlp_dim
                ),
                # activation.__dict__[hidden_activation](
                #         **self.hidden_activation_kwargs
                # ),
        )

        self.output = (
                nn.Sequential(
                        nn.Linear(
                                in_features=mlp_dim,
                                out_features=bandwidth * in_channel * self.reim * self.glu_mult,
                        ),
                        nn.GLU(dim=-1),
                )
        )


        if conv_kernel_freq is not None:
            warnings.warn(
                    "`conv_kernel_freq` is not None, but it is not used in `KernelNormMLP3`. "
            )

    def reshape_output(self, mb):

        # mb: (batch, n_time, mask_kernel_freq, mask_kernel_time, bandwidth * in_channel * reim)
        # output: (batch, in_channel, kernel_freq, kernel_time, bandwidth, n_time)

        batch, n_time, _, _, _ = mb.shape
        if self.complex_mask:
            mb = mb.reshape(
                    batch,
                    n_time,
                    self.mask_kernel_freq,
                    self.mask_kernel_time,
                    self.bandwidth,
                    self.in_channel,
                    self.reim,
            ).contiguous()
            mb = torch.view_as_complex(mb)
        else:
            mb = mb.reshape(
                    batch,
                    n_time,
                    self.mask_kernel_freq,
                    self.mask_kernel_time,
                    self.bandwidth,
                    self.in_channel,
            )

        mb = torch.permute(mb, (0, 5, 2, 3, 4, 1))

        return mb

    def forward(self, qb):
        qb = self.norm(qb)  # (batch, n_time, emb_dim)
        qb = self.hidden(qb)  # (batch, n_time, mlp_dim)
        qb = torch.permute(qb, (0, 2, 1))  # (batch, mlp_dim, n_time)
        qb = self.hidden_conv(qb) # (batch, mlp_dim * kernel_freq * kernel_time, n_time)
        qb = torch.permute(qb, (0, 2, 1))  # (batch, n_time, mlp_dim * kernel_freq * kernel_time)
        batch, n_time, _ = qb.shape
        qb = qb.reshape(
                batch,
                n_time,
                self.mask_kernel_freq,
                self.mask_kernel_time,
                -1,
        )

        mb = self.output(qb) # (batch, n_time, mask_kernel_freq, mask_kernel_time, bandwidth * in_channel * reim)
        mb = self.reshape_output(mb)

        return mb

class MaskEstimationModuleSuperBase(nn.Module):
    pass


class MaskEstimationModuleBase(MaskEstimationModuleSuperBase):
    def __init__(
            self,
            band_specs: List[Tuple[float, float]],
            emb_dim: int,
            mlp_dim: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Dict = None,
            complex_mask: bool = True,
            norm_mlp_cls: Type[nn.Module] = NormMLP,
            norm_mlp_kwargs: Dict = None,
    ) -> None:
        super().__init__()

        self.band_widths = band_widths_from_specs(band_specs)
        self.n_bands = len(band_specs)

        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        if norm_mlp_kwargs is None:
            norm_mlp_kwargs = {}

        self.norm_mlp = nn.ModuleList(
                [
                        (
                                norm_mlp_cls(
                                        bandwidth=self.band_widths[b],
                                        emb_dim=emb_dim,
                                        mlp_dim=mlp_dim,
                                        in_channel=in_channel,
                                        hidden_activation=hidden_activation,
                                        hidden_activation_kwargs=hidden_activation_kwargs,
                                        complex_mask=complex_mask,
                                        **norm_mlp_kwargs,
                                )
                        )
                        for b in range(self.n_bands)
                ]
        )

    def compute_masks(self, q):
        batch, n_bands, n_time, emb_dim = q.shape

        masks = []

        for b, nmlp in enumerate(self.norm_mlp):
            # print(f"maskestim/{b:02d}")
            qb = q[:, b, :, :]
            mb = nmlp(qb)
            masks.append(mb)

        return masks



class OverlappingMaskEstimationModule(MaskEstimationModuleBase):
    def __init__(
            self,
            in_channel: int,
            band_specs: List[Tuple[float, float]],
            freq_weights: List[torch.Tensor],
            n_freq: int,
            emb_dim: int,
            mlp_dim: int,
            cond_dim: int = 0,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Dict = None,
            complex_mask: bool = True,
            norm_mlp_cls: Type[nn.Module] = NormMLP,
            norm_mlp_kwargs: Dict = None,
            use_freq_weights: bool = True,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)

        # if cond_dim > 0:
        #     raise NotImplementedError

        super().__init__(
                band_specs=band_specs,
                emb_dim=emb_dim + cond_dim,
                mlp_dim=mlp_dim,
                in_channel=in_channel,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                norm_mlp_cls=norm_mlp_cls,
                norm_mlp_kwargs=norm_mlp_kwargs,
        )

        self.n_freq = n_freq
        self.band_specs = band_specs
        self.in_channel = in_channel

        if freq_weights is not None:
            for i, fw in enumerate(freq_weights):
                self.register_buffer(f"freq_weights/{i}", fw)

                self.use_freq_weights = use_freq_weights
        else:
            self.use_freq_weights = False

        self.cond_dim = cond_dim

    def forward(self, q, cond=None):
        # q = (batch, n_bands, n_time, emb_dim)

        batch, n_bands, n_time, emb_dim = q.shape

        if cond is not None:
            print(cond)
            if cond.ndim == 2:
                cond = cond[:, None, None, :].expand(-1, n_bands, n_time, -1)
            elif cond.ndim == 3:
                assert cond.shape[1] == n_time
            else:
                raise ValueError(f"Invalid cond shape: {cond.shape}")

            q = torch.cat([q, cond], dim=-1)
        elif self.cond_dim > 0:
            cond = torch.ones(
                    (batch, n_bands, n_time, self.cond_dim),
                    device=q.device,
                    dtype=q.dtype,
            )
            q = torch.cat([q, cond], dim=-1)
        else:
            pass

        mask_list = self.compute_masks(
                q
        )  # [n_bands  * (batch, in_channel, bandwidth, n_time)]

        masks = torch.zeros(
                (batch, self.in_channel, self.n_freq, n_time),
                device=q.device,
                dtype=mask_list[0].dtype,
        )

        for im, mask in enumerate(mask_list):
            fstart, fend = self.band_specs[im]
            if self.use_freq_weights:
                fw = self.get_buffer(f"freq_weights/{im}")[:, None]
                mask = mask * fw
            masks[:, :, fstart:fend, :] += mask

        return masks



class MultAddMaskEstimationModule(MaskEstimationModuleBase):
    def __init__(
            self,
            in_channel: int,
            band_specs: List[Tuple[float, float]],
            freq_weights: List[torch.Tensor],
            n_freq: int,
            emb_dim: int,
            mlp_dim: int,
            cond_dim: int = 0,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Dict = None,
            complex_mask: bool = True,
            norm_mlp_cls: Type[nn.Module] = MultAddNormMLP,
            norm_mlp_kwargs: Dict = None,
            use_freq_weights: bool = True,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)
        super().__init__(
                band_specs=band_specs,
                emb_dim=emb_dim + cond_dim,
                mlp_dim=mlp_dim,
                in_channel=in_channel,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                norm_mlp_cls=norm_mlp_cls,
                norm_mlp_kwargs=norm_mlp_kwargs,
        )

        self.n_freq = n_freq
        self.band_specs = band_specs
        self.in_channel = in_channel

        if freq_weights is not None:
            for i, fw in enumerate(freq_weights):
                self.register_buffer(f"freq_weights/{i}", fw)

                self.use_freq_weights = use_freq_weights
        else:
            self.use_freq_weights = False

        self.cond_dim = cond_dim

    def forward(self, q, cond : torch.Tensor= None):
        # q = (batch, n_bands, n_time, emb_dim)
        # c = (batch, cond_dim) or (batch, n_time, cond_dim)

        batch, n_bands, n_time, emb_dim = q.shape

        if cond is not None:
            if cond.ndim == 2:
                cond = cond[:, None, None, :].expand(-1, n_bands, n_time, -1)
            elif cond.ndim == 3:
                assert cond.shape[1] == n_time
            else:
                raise ValueError(f"Invalid cond shape: {cond.shape}")

            q = torch.cat([q, cond], dim=-1)
        elif self.cond_dim > 0:
            cond = torch.ones(
                    (batch, n_bands, n_time, self.cond_dim),
                    device=q.device,
                    dtype=q.dtype,
            )
            q = torch.cat([q, cond], dim=-1)
        else:
            pass

        mask_list = self.compute_masks(
                q
        )  # [n_bands  * (batch, in_channel, bandwidth, n_time)]

        mmasks = torch.zeros(
                (batch, self.in_channel, self.n_freq, n_time),
                device=q.device,
                dtype=mask_list[0][0].dtype,
        )

        amasks = torch.zeros(
                (batch, self.in_channel, self.n_freq, n_time),
                device=q.device,
                dtype=mask_list[0][0].dtype,
        )

        for im, (mmask, amask) in enumerate(mask_list):
            fstart, fend = self.band_specs[im]
            if self.use_freq_weights:
                fw = self.get_buffer(f"freq_weights/{im}")[:, None]
                mmask = mmask * fw
                amask = amask * fw
            mmasks[:, :, fstart:fend, :] += mmask
            amasks[:, :, fstart:fend, :] += amask

        # print(mmasks.shape, amasks.shape)

        return torch.stack([mmasks, amasks], dim=-1)

class MaskEstimationModule(OverlappingMaskEstimationModule):
    def __init__(
            self,
            band_specs: List[Tuple[float, float]],
            emb_dim: int,
            mlp_dim: int,
            in_channel: Optional[int],
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Dict = None,
            complex_mask: bool = True,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)
        check_no_overlap(band_specs)
        super().__init__(
                in_channel=in_channel,
                band_specs=band_specs,
                freq_weights=None,
                n_freq=None,
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
        )

    def forward(self, q):
        # q = (batch, n_bands, n_time, emb_dim)

        masks = self.compute_masks(
                q
        )  # [n_bands  * (batch, in_channel, bandwidth, n_time)]

        # TODO: currently this requires band specs to have no gap and no overlap
        masks = torch.concat(
                masks,
                dim=2
        )  # (batch, in_channel, n_freq, n_time)

        return masks


class PatchingMaskEstimationModule(OverlappingMaskEstimationModule):
    def __init__(
            self,
            in_channel: int,
            band_specs: List[Tuple[float, float]],
            freq_weights: List[torch.Tensor],
            n_freq: int,
            emb_dim: int,
            mlp_dim: int,
            mask_kernel_freq: int,
            mask_kernel_time: int,
            conv_kernel_freq: int,
            conv_kernel_time: int,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Dict = None,
            complex_mask: bool = True,
            kernel_norm_mlp_version: int = 1,
    ) -> None:

        if kernel_norm_mlp_version == 1:
            norm_mlp_cls = KernelNormMLP
        elif kernel_norm_mlp_version == 2:
            norm_mlp_cls = KernelNormMLP2
        elif kernel_norm_mlp_version == 3:
            norm_mlp_cls = KernelNormMLP3
        else:
            raise ValueError(f"Invalid kernel_norm_mlp_version: {kernel_norm_mlp_version}")


        super().__init__(
                in_channel=in_channel,
                band_specs=band_specs,
                freq_weights=freq_weights,
                n_freq=n_freq,
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                norm_mlp_cls=norm_mlp_cls,
                norm_mlp_kwargs=dict(
                        mask_kernel_freq=mask_kernel_freq,
                        mask_kernel_time=mask_kernel_time,
                        conv_kernel_freq=conv_kernel_freq,
                        conv_kernel_time=conv_kernel_time,
                )
        )

        self.mask_kernel_freq = mask_kernel_freq
        self.mask_kernel_time = mask_kernel_time

    def forward(self, q):
        # q = (batch, n_bands, n_time, emb_dim)

        batch, n_bands, n_time, emb_dim = q.shape

        mask_list = self.compute_masks(q)
        # [n_bands  * (batch, in_channel, bandwidth, n_time)]

        masks = torch.zeros(
                (batch, self.in_channel, self.mask_kernel_freq,
                 self.mask_kernel_time,
                 self.n_freq, n_time),
                device=q.device,
                dtype=mask_list[0].dtype,
        )

        for im, mask in enumerate(mask_list):
            fstart, fend = self.band_specs[im]
            if self.use_freq_weights:
                fw = self.get_buffer(f"freq_weights/{im}")[:, None]
                mask = mask * fw
            masks[..., fstart:fend, :] += mask

        return masks
