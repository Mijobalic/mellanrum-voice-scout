import argparse
import logging
import queue
import sounddevice as sd
import torch
from pathlib import Path
from typing import Optional

from src.voice_scout.vad import SileroVAD
from src.voice_scout.transcriber import WhisperTranscriber
from src.voice_scout.cortex_client import CortexClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceScout:
    def __init__(
        self,
        model_path: Path = Path("models/silero_vad.onnx"),
        whisper_model: str = "base",
        cortex_url: str = "http://localhost:8000",
        sample_rate: int = 16000,
        device: str = "cpu"
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.vad = SileroVAD(model_path, device)
        self.transcriber = WhisperTranscriber(model_size=whisper_model, device=device)
        self.cortex_client = CortexClient(cortex_url)

    def _audio_callback(self, indata: ndarray, frames: int, time, status: sd.CallbackFlags):
        if status:
            logger.warning(f"Audio stream status: {status}")

        # Convert to mono if needed and ensure correct sample rate
        if indata.ndim > 1:
            indata = indata.mean(axis=1)
        if indata.shape[0] == 0:
            return

        # Process with VAD
        is_speech = self.vad.is_speech(indata)
        if is_speech:
            self.transcriber.add_audio(indata)

    def start(self):
        logger.info("Starting Voice Scout...")
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback
        )
        stream.start()

        try:
            while True:
                # Check for completed transcriptions
                transcription = self.transcriber.get_transcription()
                if transcription:
                    logger.info(f"Transcription: {transcription}")
                    self.cortex_client.send_transcript(transcription)
        except KeyboardInterrupt:
            logger.info("Stopping Voice Scout...")
            stream.stop()
            stream.close()

def main():
    parser = argparse.ArgumentParser(description="Mellanrum Voice Scout")
    parser.add_argument("--model-path", type=Path, default="models/silero_vad.onnx",
                       help="Path to Silero VAD model")
    parser.add_argument("--whisper-model", type=str, default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--cortex-url", type=str, default="http://localhost:8000",
                       help="URL of Mellanrum Cortex")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="Audio sample rate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to use for processing")

    args = parser.parse_args()

    scout = VoiceScout(
        model_path=args.model_path,
        whisper_model=args.whisper_model,
        cortex_url=args.cortex_url,
        sample_rate=args.sample_rate,
        device=args.device
    )
    scout.start()

if __name__ == "__main__":
    main()
