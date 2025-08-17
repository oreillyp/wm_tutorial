import math
import torch
import copy


from numpy.random import RandomState
from dasp_pytorch.functional import noise_shaped_reverberation

from audiotools import AudioSignal, STFTParams
from audiotools.core.util import sample_from_dist
from audiotools.data.transforms import BaseTransform


class Noise(BaseTransform):
    def __init__(
        self, 
        snr: tuple = ("uniform", 10.0, 30.0),
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)
        
        self.snr = snr
        self.eq_amount = eq_amount
        self.n_bands = n_bands

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        eq_amount = sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        snr = sample_from_dist(self.snr, state)

        bg_signal = signal[0].clone()
        bg_signal.audio_data = torch.randn_like(bg_signal.audio_data)
        bg_signal.ensure_max_of_audio()

        return {"eq": eq, "bg_signal": bg_signal, "snr": snr}

    def _transform(self, signal, bg_signal, snr, eq):
        return signal.mix(bg_signal.clone(), snr, eq)


class Reverb(BaseTransform):
    """
    Wraps filtered noise reverb from dasp-pytorch; see
    https://github.com/csteinmetz1/dasp-pytorch/blob/main/dasp_pytorch/functional.py
    """
    def __init__(
        self, 
        n_taps: int = 1023,
        n_samples: int = 65536,
        snr: tuple = ("uniform", 0.0, 30.0),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.n_taps = n_taps
        self.n_samples = n_samples
        self.snr = snr

    def _instantiate(self, state: RandomState, signal: AudioSignal):

        reverb_kwargs = {}

        # Sample 12 random band gains in [0, 1]
        for i in range(12):
            reverb_kwargs[f"band{i}_gain"] = sample_from_dist(("uniform", 0.0, 1.0))
            
        # Sample 12 random band decays in [0, 1]
        for i in range(12):
            reverb_kwargs[f"band{i}_decay"] = sample_from_dist(("uniform", 0.0, 1.0))
        
        # Sample SNR
        snr = sample_from_dist(self.snr, state)
        reverb_kwargs["snr"] = snr

        return reverb_kwargs

    def _transform(
        self, 
        signal,
        snr,
        band0_gain: torch.Tensor,
        band1_gain: torch.Tensor,
        band2_gain: torch.Tensor,
        band3_gain: torch.Tensor,
        band4_gain: torch.Tensor,
        band5_gain: torch.Tensor,
        band6_gain: torch.Tensor,
        band7_gain: torch.Tensor,
        band8_gain: torch.Tensor,
        band9_gain: torch.Tensor,
        band10_gain: torch.Tensor,
        band11_gain: torch.Tensor,
        band0_decay: torch.Tensor,
        band1_decay: torch.Tensor,
        band2_decay: torch.Tensor,
        band3_decay: torch.Tensor,
        band4_decay: torch.Tensor,
        band5_decay: torch.Tensor,
        band6_decay: torch.Tensor,
        band7_decay: torch.Tensor,
        band8_decay: torch.Tensor,
        band9_decay: torch.Tensor,
        band10_decay: torch.Tensor,
        band11_decay: torch.Tensor,
    ):

        out = signal.clone().resample(44_100)

        # Convert SNR to linear mixing factor
        snr_lin = torch.pow(10.0, snr / 10)
        mix = snr_lin / (1 + snr_lin)
        mix.clamp_(0.0, 1.0)

        orig_n_channels = out.num_channels
        if orig_n_channels == 1:
            out.audio_data = out.audio_data.repeat(1, 2, 1)
        
        out.audio_data = noise_shaped_reverberation(
            x=out.audio_data,
            sample_rate=out.sample_rate,
            num_samples=self.n_samples,
            num_bandpass_taps=self.n_taps,
            mix=mix.unsqueeze(-1).to(out.device),
            band0_gain=band0_gain.unsqueeze(-1).to(out.device),
            band1_gain=band1_gain.unsqueeze(-1).to(out.device),
            band2_gain=band2_gain.unsqueeze(-1).to(out.device),
            band3_gain=band3_gain.unsqueeze(-1).to(out.device),
            band4_gain=band4_gain.unsqueeze(-1).to(out.device),
            band5_gain=band5_gain.unsqueeze(-1).to(out.device),
            band6_gain=band6_gain.unsqueeze(-1).to(out.device),
            band7_gain=band7_gain.unsqueeze(-1).to(out.device),
            band8_gain=band8_gain.unsqueeze(-1).to(out.device),
            band9_gain=band9_gain.unsqueeze(-1).to(out.device),
            band10_gain=band10_gain.unsqueeze(-1).to(out.device),
            band11_gain=band11_gain.unsqueeze(-1).to(out.device),
            band0_decay=band0_decay.unsqueeze(-1).to(out.device),
            band1_decay=band1_decay.unsqueeze(-1).to(out.device),
            band2_decay=band2_decay.unsqueeze(-1).to(out.device),
            band3_decay=band3_decay.unsqueeze(-1).to(out.device),
            band4_decay=band4_decay.unsqueeze(-1).to(out.device),
            band5_decay=band5_decay.unsqueeze(-1).to(out.device),
            band6_decay=band6_decay.unsqueeze(-1).to(out.device),
            band7_decay=band7_decay.unsqueeze(-1).to(out.device),
            band8_decay=band8_decay.unsqueeze(-1).to(out.device),
            band9_decay=band9_decay.unsqueeze(-1).to(out.device),
            band10_decay=band10_decay.unsqueeze(-1).to(out.device),
            band11_decay=band11_decay.unsqueeze(-1).to(out.device),
        )

        if orig_n_channels == 1:
            out = out.to_mono()

        out.resample(signal.sample_rate)
        out = out[..., :signal.signal_length]
        out.audio_data = torch.nn.functional.pad(
            out.audio_data,
            (0, max(0, signal.signal_length - out.signal_length))
        )
        
        return out


class Speed(BaseTransform):

    def __init__(
        self, 
        factor: tuple = ("choice", (0.99, 1.01)),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)
        
        self.factor = factor

    def _instantiate(self, state: RandomState):

        factor = sample_from_dist(self.factor, state)
        return {"factor": factor}

    def _transform(self, signal, factor):

        if isinstance(factor, torch.Tensor):
            factor = factor.tolist()
        elif isinstance(factor, (float, int)):
            factor = [factor]

        out = signal.clone()
        
        for i, _factor in enumerate(factor):

            src_rate = int(_factor * signal.sample_rate)
            tgt_rate = int(signal.sample_rate)
            
            # Keep GCD of source and target sample rates reasonably large to 
            # limit resampling kernel size
            assert not tgt_rate % 50
            src_rate = (src_rate // 50 * 50)

            _out = out[i].clone()
            _out.sample_rate = src_rate
            _out = _out.resample(tgt_rate)
            _len = min(out.shape[-1], _out.shape[-1])
            out.audio_data[i, :, :_len] = _out.audio_data[..., :_len]

        return out