import torch
from audiotools import AudioSignal, STFTParams


def _dct_1d(x: torch.Tensor, norm: str = "ortho"):
    n_samples = x.shape[-1]
    x_ext = torch.cat([x, x.flip(dims=[-1])], dim=-1)   # (..., 2*n_samples)
    X = torch.fft.fft(x_ext, dim=-1)

    k = torch.arange(n_samples, device=x.device, dtype=x.dtype)
    twiddle = torch.exp(-1j * torch.pi * k / (2 * n_samples)).to(X.dtype)

    C = 0.5 * (X[..., :n_samples] * twiddle).real

    if norm == "ortho":
        C[..., 0] *= (1.0 / torch.sqrt(torch.tensor(n_samples, device=x.device, dtype=x.dtype)))
        C[..., 1:] *= (torch.sqrt(torch.tensor(2.0 / n_samples, device=x.device, dtype=x.dtype)))
    else:
        C[..., 1:] *= 2.0
    return C


def _idct_1d(X: torch.Tensor, norm: str = "ortho"):

    # Inverse scaling to match _dct_1d "ortho" behavior.
    n_samples = X.shape[-1]
    x = X.clone()

    if norm == "ortho":
        x[..., 0] *= (1.0 / torch.sqrt(torch.tensor(n_samples, device=X.device, dtype=X.dtype)))
        x[..., 1:] *= (torch.sqrt(torch.tensor(2.0 / n_samples, device=X.device, dtype=X.dtype)))
    else:
        x[..., 0] *= 0.5
        x *= 2.0

    # Direct DCT-III via matrix multiplication
    k = torch.arange(n_samples, device=X.device, dtype=X.dtype)
    n = k.view(-1, 1)  # (L, 1)
    cos = torch.cos(torch.pi * (2 * n + 1) * k / (2 * n_samples))  # (L, L)
    return x @ cos.T


def dct(
    signal: AudioSignal,
    window_length: int = None,
    hop_length: int = None,
    window_type: str = None,
    match_stride: bool = None,
    padding_type: str = None,
    norm: str = "ortho",
):
    """
    Frame-wise DCT (DCT-II) over last dimension of signal.audio_data
    """
    
    # Resolve params from signal.stft_params for parity with .stft()
    window_length = (
        signal.stft_params.window_length if window_length is None else int(window_length)
    )
    hop_length = signal.stft_params.hop_length if hop_length is None else int(hop_length)
    window_type = signal.stft_params.window_type if window_type is None else window_type
    match_stride = (
        signal.stft_params.match_stride if match_stride is None else bool(match_stride)
    )
    padding_type = (
        signal.stft_params.padding_type if padding_type is None else padding_type
    )

    # Get analysis window
    window = signal.get_window(window_type, window_length, signal.audio_data.device)

    # Pad
    audio = signal.audio_data
    right_pad, pad = signal.compute_stft_padding(window_length, hop_length, match_stride)
    audio = torch.nn.functional.pad(audio, (pad, pad + right_pad), padding_type)

    # Flatten batch/channels
    n_batch, n_channels, n_samples = audio.shape
    x = audio.reshape(-1, n_samples)  # (n_batch*n_channels, n_samples)

    # Frame to (N, n_frames, window_length)
    n_frames = 1 + (n_samples - window_length) // hop_length
    frames = x.unfold(dimension=1, size=window_length, step=hop_length)  # (N, n_frames, win)
    if match_stride:
        # Mirror STFT behavior: drop first two and last two frames (added by padding)
        if frames.shape[1] >= 4:
            frames = frames[:, 2:-2, :]

    # Apply window
    frames = frames * window.view(1, 1, -1)

    # DCT along window axis
    coeffs = _dct_1d(frames, norm=norm)  # (N, n_frames, win)

    # Return as (n_batch, n_channels, n_freq, n_frames) to match STFT layout
    N = n_batch * n_channels
    _, n_frames_eff, n_freq = coeffs.shape
    coeffs = coeffs.reshape(n_batch, n_channels, n_frames_eff, n_freq).permute(0, 1, 3, 2).contiguous()
    return coeffs


def idct(
    signal: AudioSignal,
    dct_data: torch.Tensor,          # (n_batch, n_channels, n_freq, n_frames)
    window_length: int = None,
    hop_length: int = None,
    window_type: str = None,
    match_stride: bool = None,
    length: int = None,
    norm: str = "ortho",
):
    """
    Overlap-add DCT-III (inverse of DCT-II). Returns the same AudioSignal with
    audio_data replaced by the reconstruction.

    Notes:
      * Uses window-squared normalization for perfect OLA (independent of COLA).
      * `length` defaults to the same behavior as your iSTFT: original length
        plus pad bookkeeping, then cropped if match_stride=True.
    """
    # Resolve params
    window_length = (
        signal.stft_params.window_length if window_length is None else int(window_length)
    )
    hop_length = signal.stft_params.hop_length if hop_length is None else int(hop_length)
    window_type = signal.stft_params.window_type if window_type is None else window_type
    match_stride = (
        signal.stft_params.match_stride if match_stride is None else bool(match_stride)
    )

    # Synthesis window
    window = signal.get_window(window_type, window_length, dct_data.device)

    n_batch, n_channels, n_freq, n_frames = dct_data.shape
    assert n_freq == window_length, "dct_data n_freq must equal window_length"

    # Put frames back to (N, n_frames, win)
    frames_D = dct_data.permute(0, 1, 3, 2).contiguous().view(n_batch * n_channels, n_frames, n_freq)

    # If match_stride, restore the two frames on either side that analysis dropped
    right_pad, pad = signal.compute_stft_padding(window_length, hop_length, match_stride)
    if match_stride:
        frames_D = torch.nn.functional.pad(frames_D, (0, 0, 2, 2))

    # Inverse DCT (along window axis)
    time_frames = _idct_1d(frames_D, norm=norm)  # (N, n_frames_eff, win)

    # Apply synthesis window
    time_frames = time_frames * window.view(1, 1, -1)

    # Overlap-add
    N, nf_eff, win = time_frames.shape
    total_len = (nf_eff - 1) * hop_length + win
    y = time_frames.new_zeros(N, total_len)
    wsum = time_frames.new_zeros(N, total_len)

    for i in range(nf_eff):
        start = i * hop_length
        end = start + win
        y[:, start:end] += time_frames[:, i, :]
        wsum[:, start:end] += window.pow(2).view(1, -1)

    # Normalize where window overlaps
    eps = torch.finfo(y.dtype).eps
    y = y / (wsum + eps)

    # Determine target length following your iSTFT conventions
    if length is None:
        length = signal.signal_length
        length = length + 2 * pad + right_pad

    # Crop to requested length
    if y.size(-1) < length:
        y = torch.nn.functional.pad(y, (0, length - y.size(-1)))
    else:
        y = y[..., :length]

    # If match_stride: remove the analysis padding
    if match_stride:
        y = y[..., pad : (y.size(-1) - (pad + right_pad))]

    # Reshape back to (n_batch, n_channels, n_samples)
    audio = y.view(n_batch, n_channels, -1)
    signal.audio_data = audio
    return signal
