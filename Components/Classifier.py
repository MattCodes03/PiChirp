import numpy as np
import librosa
import torch
import time

class Classifier:
	def __init__(self, model):
		self.model = model
		self.buffer = []


	def process_audio(self, audio_chunk):
		audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)

        # Handle multi-channel audio by averaging channels (if stereo)
		if len(audio_data.shape) > 1:
			audio_data = np.mean(audio_data, axis=1)

        # Normalize the audio data
		audio_data /= np.max(np.abs(audio_data) + 1e-9)

		max_frames = 313

		mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=44100, n_mels=64)
		mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

		# Normalize and pad/truncate spectrogram to fixed size
		mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

		delta = librosa.feature.delta(mel_spec)
		delta2 = librosa.feature.delta(mel_spec, order=2)

		# Normalize delta features
		delta = (delta - delta.min()) / (delta.max() - delta.min())
		delta2 = (delta2 - delta2.min()) / (delta2.max() - delta2.min())
		
		mel_spec = np.concatenate([mel_spec, delta, delta2], axis=0)

		if mel_spec.shape[1] < max_frames:
			pad_width = max_frames - mel_spec.shape[1]
			mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="constant")
		else:
			mel_spec = mel_spec[:, :max_frames]
		
		mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
		return mel_spec

	def __call__(self, audio_chunk):
		print("Classification Started!")
		mel_spec = self.process_audio(audio_chunk)

		start_time = time.time()

		with torch.no_grad():
			output = self.model(mel_spec)
			probability = torch.nn.functional.softmax(output, dim=1).squeeze(0).numpy()

		end_time = time.time()
		inference_time = end_time - start_time

		# Find the class with the highest probability
		max_prob = np.max(probability)
		index = np.argmax(probability)

		return index, max_prob, inference_time
