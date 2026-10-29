# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------
"""Vectorized STFT helpers shared by the audio representations.

librosa rebuilds the window function and the mel filterbank on every call,
and its stft has a large fixed per-call cost. Context operators call the
audio representations once per batch of windows, so that overhead dominated
their runtime. These helpers cache the window and filterbank and transform
any leading batch shape with one FFT call. Results match librosa's defaults
(Hann window, center=True, zero padding) up to float rounding.
"""

import functools

import librosa
import numpy as np
import scipy.fft
from numpy.lib.stride_tricks import sliding_window_view

# Bound on the framed-signal elements materialized at once, so long signals
# do not need a full (frames x n_fft) copy next to the spectrogram.
_MAX_BLOCK_ELEMENTS = 1 << 22


@functools.lru_cache(maxsize=None)
def hann_window(n_fft, dtype):
    window = librosa.filters.get_window("hann", n_fft, fftbins=True).astype(dtype)
    window.setflags(write=False)
    return window


@functools.lru_cache(maxsize=64)
def mel_basis(sr, n_fft, n_mels):
    basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    basis.setflags(write=False)
    return basis


def _framed_power(frames, window):
    spectrum = scipy.fft.rfft(frames * window, axis=-1)
    return np.square(spectrum.real) + np.square(spectrum.imag)


def power_spectrogram(y, n_fft, hop_length):
    """|STFT|^2 over the last axis of y, shaped (..., frames, 1 + n_fft // 2)."""
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32)
    pad = n_fft // 2
    y = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(pad, pad)])
    frames = sliding_window_view(y, n_fft, axis=-1)[..., ::hop_length, :]
    window = hann_window(n_fft, y.dtype)

    num_frames = frames.shape[-2]
    batch_size = int(np.prod(frames.shape[:-2], dtype=np.int64))
    block = max(1, _MAX_BLOCK_ELEMENTS // max(1, batch_size * n_fft))
    if block >= num_frames:
        return _framed_power(frames, window)

    power = np.empty((*frames.shape[:-1], n_fft // 2 + 1), dtype=y.dtype)
    for start in range(0, num_frames, block):
        power[..., start : start + block, :] = _framed_power(
            frames[..., start : start + block, :], window
        )
    return power


def mel_spectrogram(y, sr, n_fft, hop_length, n_mels):
    """Power mel spectrogram over the last axis of y, shaped (..., frames, n_mels)."""
    power = power_spectrogram(y, n_fft, hop_length)
    return power @ mel_basis(sr, n_fft, n_mels).T


def power_to_db(S, amin=1e-10, top_db=80.0):
    """librosa.power_to_db (ref=1.0) with top_db applied per spectrogram.

    librosa clips against the maximum of the whole array, which would make
    each window's features depend on the other windows of its batch. Here
    the last two axes form one spectrogram.
    """
    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    if top_db is not None:
        peak = log_spec.max(axis=(-2, -1), keepdims=True)
        np.maximum(log_spec, peak - top_db, out=log_spec)
    return log_spec
