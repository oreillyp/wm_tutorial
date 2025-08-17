import math
import torch
import copy

from typing import List
from numpy.random import RandomState

from audiotools import AudioSignal, STFTParams
from audiotools.core.util import sample_from_dist
from audiotools.data.transforms import BaseTransform
from audiotools.data.datasets import AudioLoader


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
    Patch device error in `audiotools.data.transforms.RoomImpulseResponse` to 
    allow training on GPU.
    """

    def __init__(
        self,
        drr: tuple = ("uniform", 0.0, 30.0),
        sources: List[str] = None,
        weights: List[float] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        name: str = None,
        prob: float = 1.0,
        use_original_phase: bool = False,
        offset: float = 0.0,
        duration: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.drr = drr
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.use_original_phase = use_original_phase

        self.loader = AudioLoader(sources, weights)
        self.offset = offset
        self.duration = duration

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        eq_amount = sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        drr = sample_from_dist(self.drr, state)

        ir_signal = self.loader(
            state,
            signal.sample_rate,
            offset=self.offset,
            duration=self.duration,
            loudness_cutoff=None,
            num_channels=signal.num_channels,
        )["signal"]
        ir_signal.zero_pad_to(signal.sample_rate)

        return {"eq": eq, "ir_signal": ir_signal, "drr": drr}

    def _transform(self, signal, ir_signal, drr, eq):

        if isinstance(drr, torch.Tensor):
            drr = drr.to(signal.device)
        if isinstance(eq, torch.Tensor):
            eq = eq.to(signal.device)
        ir_signal = ir_signal.clone().to(signal.device)
        
        return signal.apply_ir(
            ir_signal, 
            drr, 
            eq, 
            use_original_phase=self.use_original_phase
        )


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