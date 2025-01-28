import pyaudio
import sounddevice as sd
import wave
import time
import librosa
import torch
from datetime import datetime
import numpy as np
import queue
import threading
import os

class AudioRecorder:
    def __init__(self, channels=1, sample_rate=44100, dtype='int16'):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype

    def record_audio(self, duration=10):
        """Records audio for a specified duration (in seconds)."""
        print(f"Recording audio for {duration} seconds...")
        audio_data = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate,
                            channels=self.channels, dtype=self.dtype)
        sd.wait()  # Wait until recording is finished
        print("Recording complete.")
        return audio_data.tobytes()
