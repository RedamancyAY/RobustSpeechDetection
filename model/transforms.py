# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import math
import numbers
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from hide_warnings import hide_warnings
from torchaudio.io import AudioEffector

# -

class SpecAugmentTransform:
    """SpecAugment (https://arxiv.org/abs/1904.08779)"""

    @classmethod
    def _set_specaugment(
        cls,
        time_warp_w: int,
        freq_mask_n: int,
        freq_mask_f: int,
        time_mask_n: int,
        time_mask_t: int,
        time_mask_p: float,
    ):
        return {
            "time_warp_W": time_warp_w,
            "freq_mask_N": freq_mask_n,
            "freq_mask_F": freq_mask_f,
            "time_mask_N": time_mask_n,
            "time_mask_T": time_mask_t,
            "time_mask_p": time_mask_p,
        }

    @classmethod
    def from_policy(cls, policy):
        POLICYS = {
            "lb": cls._set_specaugment(
                time_warp_w=0,
                freq_mask_n=1,
                freq_mask_f=27,
                time_mask_n=1,
                time_mask_t=100,
                time_mask_p=1.0,
            ),
            "ld": cls._set_specaugment(
                time_warp_w=0,
                freq_mask_n=2,
                freq_mask_f=27,
                time_mask_n=2,
                time_mask_t=100,
                time_mask_p=1.0,
            ),
            "sm": cls._set_specaugment(
                time_warp_w=0,
                freq_mask_n=2,
                freq_mask_f=15,
                time_mask_n=2,
                time_mask_t=70,
                time_mask_p=0.2,
            ),
            "ss": cls._set_specaugment(
                time_warp_w=0,
                freq_mask_n=2,
                freq_mask_f=27,
                time_mask_n=2,
                time_mask_t=70,
                time_mask_p=0.2,
            ),
            "ay": cls._set_specaugment(
                time_warp_w=0,
                freq_mask_n=2,
                freq_mask_f=27,
                time_mask_n=2,
                time_mask_t=70,
                time_mask_p=0.5,
            ),
        }
        assert policy in POLICYS.keys()
        config = POLICYS[policy]
        return cls.from_config_dict(config)

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return SpecAugmentTransform(
            _config.get("time_warp_W", 0),
            _config.get("freq_mask_N", 0),
            _config.get("freq_mask_F", 0),
            _config.get("time_mask_N", 0),
            _config.get("time_mask_T", 0),
            _config.get("time_mask_p", 0.0),
            _config.get("mask_value", None),
        )

    def __init__(
        self,
        time_warp_w: int = 0,
        freq_mask_n: int = 0,
        freq_mask_f: int = 0,
        time_mask_n: int = 0,
        time_mask_t: int = 0,
        time_mask_p: float = 0.0,
        mask_value: Optional[float] = 0.0,
    ):
        # Sanity checks
        assert mask_value is None or isinstance(
            mask_value, numbers.Number
        ), f"mask_value (type: {type(mask_value)}) must be None or a number"
        if freq_mask_n > 0:
            assert freq_mask_f > 0, (
                f"freq_mask_F ({freq_mask_f}) "
                f"must be larger than 0 when doing freq masking."
            )
        if time_mask_n > 0:
            assert time_mask_t > 0, (
                f"time_mask_T ({time_mask_t}) must be larger than 0 when "
                f"doing time masking."
            )

        self.time_warp_w = time_warp_w
        self.freq_mask_n = freq_mask_n
        self.freq_mask_f = freq_mask_f
        self.time_mask_n = time_mask_n
        self.time_mask_t = time_mask_t
        self.time_mask_p = time_mask_p
        self.mask_value = mask_value

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                [
                    f"time_warp_w={self.time_warp_w}",
                    f"freq_mask_n={self.freq_mask_n}",
                    f"freq_mask_f={self.freq_mask_f}",
                    f"time_mask_n={self.time_mask_n}",
                    f"time_mask_t={self.time_mask_t}",
                    f"time_mask_p={self.time_mask_p}",
                ]
            )
            + ")"
        )

    def __call__(self, spectrogram):
        assert len(spectrogram.shape) == 2, "spectrogram must be a 2-D tensor."

        distorted = spectrogram.copy()  # make a copy of input spectrogram.
        num_frames = spectrogram.shape[0]  # or 'tau' in the paper.
        num_freqs = spectrogram.shape[1]  # or 'miu' in the paper.
        mask_value = self.mask_value

        if mask_value is None:  # if no value was specified, use local mean.
            mask_value = spectrogram.mean()

        if num_frames == 0:
            return spectrogram

        if num_freqs < self.freq_mask_f:
            return spectrogram

        if self.time_warp_w > 0:
            if 2 * self.time_warp_w < num_frames:
                import cv2

                w0 = np.random.randint(self.time_warp_w, num_frames - self.time_warp_w)
                w = np.random.randint(-self.time_warp_w + 1, self.time_warp_w)
                upper, lower = distorted[:w0, :], distorted[w0:, :]
                upper = cv2.resize(
                    upper, dsize=(num_freqs, w0 + w), interpolation=cv2.INTER_LINEAR
                )
                lower = cv2.resize(
                    lower,
                    dsize=(num_freqs, num_frames - w0 - w),
                    interpolation=cv2.INTER_LINEAR,
                )
                distorted = np.concatenate((upper, lower), axis=0)

        for _i in range(self.freq_mask_n):
            f = np.random.randint(0, self.freq_mask_f)
            f0 = np.random.randint(0, num_freqs - f)
            if f != 0:
                distorted[:, f0 : f0 + f] = mask_value

        max_time_mask_t = min(
            self.time_mask_t, math.floor(num_frames * self.time_mask_p)
        )
        if max_time_mask_t < 1:
            return distorted

        for _i in range(self.time_mask_n):
            t = np.random.randint(0, max_time_mask_t)
            t0 = np.random.randint(0, num_frames - t)
            if t != 0:
                distorted[t0 : t0 + t, :] = mask_value

        return distorted


class SpecAugmentBatchTransform(SpecAugmentTransform):
    """SpecAugment (https://arxiv.org/abs/1904.08779)"""

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return SpecAugmentBatchTransform(
            _config.get("time_warp_W", 0),
            _config.get("freq_mask_N", 0),
            _config.get("freq_mask_F", 0),
            _config.get("time_mask_N", 0),
            _config.get("time_mask_T", 0),
            _config.get("time_mask_p", 0.0),
            _config.get("mask_value", None),
        )
        
    
    def batch_apply(self, spectrogram):
        assert len(spectrogram.shape) == 4, "spectrogram must be a 3-D tensor (B, C, frames, freqs)."
        batch_size = spectrogram.shape[0]
        
        num_frames = spectrogram.shape[-2]  # or 'tau' in the paper.
        num_freqs = spectrogram.shape[-1]  # or 'miu' in the paper.
        mask_value = self.mask_value

        if num_frames == 0 or num_freqs < self.freq_mask_f:
            return spectrogram
        
        
        if mask_value is None:  # if no value was specified, use local mean.
            mask_value = torch.mean(spectrogram, dim=[1, 2, 3], keepdims=True)

        if self.time_warp_w > 0:
            raise NotImplementedError

        for _B in range(batch_size):
            for _i in range(self.freq_mask_n):
                f = np.random.randint(0, self.freq_mask_f)
                f0 = np.random.randint(0, num_freqs - f)
                if f != 0:
                    spectrogram[_B, :, :, f0 : f0 + f] = mask_value[_B]

        max_time_mask_t = min(
            self.time_mask_t, math.floor(num_frames * self.time_mask_p)
        )
        if max_time_mask_t < 1:
            return spectrogram

        for _B in range(batch_size):
            for _i in range(self.time_mask_n):
                t = np.random.randint(0, max_time_mask_t)
                t0 = np.random.randint(0, num_frames - t)
                if t != 0:
                    spectrogram[_B, :, t0 : t0 + t, :] = mask_value[_B]

        return spectrogram


# %%
# %load_ext autoreload
# %autoreload 2

# %%



class Torchaudio_resampler:
    def __init__(self):
        self.transforms = {}

    def function_resample(self, x, orig_freq, new_freq):
        x = torchaudio.functional.resample(
            x,
            orig_freq=orig_freq,
            new_freq=new_freq,
            resampling_method="sinc_interp_kaiser",
        )
        return x

    def transform_resample(self, x, orig_freq, new_freq):
        key = f"{orig_freq}-{new_freq}"
        if key not in self.transforms.keys():
            self.transforms[key] = T.Resample(
                orig_freq,
                new_freq,
                resampling_method="sinc_interp_kaiser",
            )

        x = self.transforms[key](x)
        return x
    
def _source_target_sample_rate(orig_freq: int, speed: float):
    source_sample_rate = int(speed * orig_freq)
    target_sample_rate = int(orig_freq)
    gcd = math.gcd(source_sample_rate, target_sample_rate)
    return source_sample_rate // gcd, target_sample_rate // gcd

@dataclass
class RandomSpeed:
    min_speed: float = 0.5
    max_speed: float = 2.0
    p: float = 0.5

    def __post_init__(self):
        """post initialization

        check the values of the min_speed and max_speed

        """
        if self.min_speed <= 0:
            raise ValueError(
                f"Error, min speed must be > 0, your input is {self.min_speed}"
            )
        if self.min_speed > self.max_speed:
            raise ValueError(
                f"Error, min_speed must < max_speed, your input is {self.min_speed} and {self.max_speed}"
            )

        self.speed_to_label = {
            x / 10: x - int(self.min_speed * 10)
            for x in range(int(self.min_speed * 10), int(self.max_speed * 10) + 1, 1)
        }

        self.resampler = Torchaudio_resampler()

    def _random_speed(self, x, speed):
        if speed == 1.0:
            return x
        else:
            orig_freq, new_freq = _source_target_sample_rate(16000, speed)

            # x, _ = speed_transform(x, 16000, speed)
            if isinstance(x, np.ndarray):
                x = librosa_resample(x, orig_freq, new_freq)
            elif isinstance(x, torch.Tensor):
                x = self.resampler.function_resample(x, orig_freq, new_freq)
                # x = self.resampler.transform_resample(x, orig_freq, new_freq)
            else:
                raise ValueError(
                    f"Error, the input audio is not np.ndarray or torch.Tensor, but is {type(x)}"
                )

            return x

    def get_random_speed(self):
        if np.random.rand() > self.p:
            target_speed = 1.0
        else:
            target_speed = np.random.rand() * (self.max_speed - self.min_speed) + self.min_speed
            target_speed = round(target_speed, 1)
        return target_speed

    def set_speed_label(self, target_speed, metadata):
        metadata["speed_label"] = self.speed_to_label[target_speed]
        metadata["speed"] = target_speed
        return metadata

    def __call__(self, x: torch.Tensor, metadata=None, **kwargs) -> torch.Tensor:
        if x.ndim not in [1, 2]:
            raise ValueError("Error, input audio should be (L), or (C, L)", x.shape)

        if np.random.rand() > self.p:
            target_speed = 1.0
        else:
            target_speed = (
                np.random.rand() * (self.max_speed - self.min_speed) + self.min_speed
            )
            target_speed = round(target_speed, 1)
        if metadata is not None:
            metadata["speed_label"] = self.speed_to_label[target_speed]
            metadata["speed"] = target_speed
        x = self._random_speed(x, target_speed)
        # print(target_speed, x.shape, self.speed_to_label[target_speed])
        return x

    def batch_apply(self, x: torch.Tensor):
        batch_size = x.shape[0]
        labels = [self.get_random_speed() for i in range(batch_size)]
        for i, speed in enumerate(labels):
            x[i] = self._random_speed(x[i], speed)
        return x, labels
    
    
@dataclass
class RandomAudioCompression:
    def __init__(self, p=0.5, sample_rate=16000):
        """post initialization

        check the values of the min_speed and max_speed

        """

        self.sample_rate = sample_rate
        self.bit_rate = [16000, 32000, 64000]
        self.format_codec = {
            "mp4": "aac",
            "ogg": "opus",
            "mp3": "mp3",
        }
        self.p = p

        self.effectors = {}
        for _format, _codec in self.format_codec.items():
            for bit_rate in self.bit_rate:
                key = self.get_setting_name(_codec, bit_rate)
                _codec = None if _codec == 'mp3' else _codec
                self.effectors[key] = AudioEffector(
                    format=_format,
                    encoder=_codec,
                    codec_config=torchaudio.io.CodecConfig(bit_rate=bit_rate),
                )

        self.setting_to_label = {
            key: i + 1 for i, key in enumerate(self.effectors.keys())
        }
        self.setting_to_label["None"] = 0

    def get_setting_name(self, codec, bit_rate):
        key = f"{codec}_{bit_rate}"
        return key

    @hide_warnings
    def _compression(self, x, compression_setting):
        if compression_setting == "None":
            return x
        x = x.transpose(0, 1) # (C, L) => (L, C)
        L = x.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self.effectors[compression_setting].apply(x, sample_rate=self.sample_rate)[:L,]
        x = x.transpose(0, 1) # (L, C) => (C, L)
        return x

    def get_random_compression_setting(self):
        if np.random.rand() > self.p:
            compression_setting = "None"
        else:
            settings = list(self.effectors.keys())
            id = int(np.random.randint(0, len(settings), 1))
            compression_setting = settings[id]
        return compression_setting

    def __call__(self, x: torch.Tensor, metadata=None, **kwargs) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("Error, input audio should be (C, L), but is ", x.shape)

        compression_setting = self.get_random_compression_setting()

        if metadata is not None:
            metadata["compression_label"] = self.setting_to_label[compression_setting]

        x = self._compression(x, compression_setting)
        return x

    def batch_apply(self, x: torch.Tensor):
        batch_size = x.shape[0]
        settings = [self.get_random_compression_setting() for _ in range(batch_size)]
        labels = [self.setting_to_label[s] for s in settings]
        for i, _setting in enumerate(settings):
            x[i] = self._random_speed(x[i], _setting)
        return x, labels


# %%
@dataclass
class RandomAudioCompressionSpeedChanging:
    def __init__(
        self,
        p_compression=0.5,
        sample_rate=16000,
        min_speed=0.5,
        max_speed=2.0,
        p_speed=1.0,
        audio_length=48000,
    ):
        """post initialization

        check the values of the min_speed and max_speed
        """

        self.compressor = RandomAudioCompression(
            p=p_compression, sample_rate=sample_rate
        )
        self.speed_changer = RandomSpeed(
            min_speed=min_speed, max_speed=max_speed, p=p_speed
        )

        self.audio_length = audio_length

    def __call__(self, x: torch.Tensor, metadata=None, **kwargs) -> torch.Tensor:

        target_speed = self.speed_changer.get_random_speed()
        
        need_length = int(self.audio_length * target_speed) + 10
        waveform_len = x.shape[1]
        if waveform_len > need_length:
            start = random.randint(0, waveform_len - need_length)
            x = x[:, start : start + need_length]

        x = self.compressor(x, metadata=metadata)
        x = self.speed_changer._random_speed(x, target_speed)
        if metadata is not None:
            metadata = self.speed_changer.set_speed_label(target_speed, metadata)
        return x

    def batch_apply(self, x: torch.Tensor):
        raise NotImplementedError
