import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Union

class SileroVAD:
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.device = device
        self.session = ort.InferenceSession(
            str(model_path),
            providers=self._get_providers()
        )
        self.state = None
        self.sr = 16000  # Expected sample rate

    def _get_providers(self):
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider"]
        return ["CPUExecutionProvider"]

    def get_speech_timestamps(self, audio: np.ndarray, sampling_rate: int = 16000) -> list:
        """Process audio and return timestamps where speech is detected"""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sampling_rate != self.sr:
            audio = self._resample(audio, sampling_rate, self.sr)

        # Pad to at least 512 samples
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)), mode='constant')

        # Process in chunks
        chunk_size = 512
        timestamps = []

        for i in range(0, len(audio) - chunk_size + 1, chunk_size // 2):
            chunk = audio[i:i+chunk_size]
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name

            if self.state is None:
                state = np.zeros((2, 1, 128), dtype=np.float32)
            else:
                state = self.state

            ort_inputs = {input_name: chunk.reshape(1, 1, -1).astype(np.float32),
                         "state": state,
                         "input_sr": np.array([sampling_rate], dtype=np.float32)}
            ort_outs = self.session.run([output_name, "state"], ort_inputs)

            speech_prob = ort_outs[0][0][0]
            self.state = ort_outs[1]

            if speech_prob > 0.5:
                start_time = i / sampling_rate
                end_time = (i + chunk_size) / sampling_rate
                timestamps.append((start_time, end_time))

        return timestamps

    def is_speech(self, audio: np.ndarray, sampling_rate: int = 16000) -> bool:
        """Quick check if audio contains speech"""
        timestamps = self.get_speech_timestamps(audio, sampling_rate)
        return len(timestamps) > 0

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling implementation"""
        if orig_sr == target_sr:
            return audio

        # Simple linear interpolation resampling
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio)
