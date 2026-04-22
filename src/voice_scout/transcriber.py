import numpy as np
import torch
from pathlib import Path
from typing import Optional
from queue import Queue
from threading import Thread

class WhisperTranscriber:
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.model = self._load_model()
        self.audio_queue = Queue()
        self.transcription_queue = Queue()
        self.running = True
        self.worker_thread = Thread(target=self._process_audio, daemon=True)
        self.worker_thread.start()

    def _load_model(self):
        try:
            model = torch.jit.load(f"models/whisper-{self.model_size}.pt")
            model.to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    def _process_audio(self):
        while self.running:
            audio_chunk = self.audio_queue.get()
            if audio_chunk is None:
                break

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_chunk).float().to(self.device)

            # Transcribe
            with torch.no_grad():
                result = self.model.transcribe(audio_tensor)

            # Put transcription in queue
            self.transcription_queue.put(result['text'])

    def add_audio(self, audio: np.ndarray):
        """Add audio chunk for transcription"""
        self.audio_queue.put(audio)

    def get_transcription(self) -> Optional[str]:
        """Get the latest transcription if available"""
        try:
            return self.transcription_queue.get_nowait()
        except:
            return None

    def __del__(self):
        self.running = False
        self.worker_thread.join()
