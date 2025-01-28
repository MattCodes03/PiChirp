from Components.AudioRecorder import AudioRecorder
from Components.Classifier import Classifier
from Components.Model import ConvLSTMModel
import torch
import psutil

print(f"Memory usage: {psutil.virtual_memory().percent}%")

recorder = AudioRecorder()

model = ConvLSTMModel()
model = torch.load('quantized_model.pth')
model.eval()
classifier = Classifier(model)

buffer = []

labels = ['Raven', 'European Starling', 'House Sparrow', 'House Wren', 'Red Crossbill']
while True:
    # Record a 10-second audio chunk
    audio_chunk = recorder.record_audio(duration=10)

    if not audio_chunk:
        print("No audio chunk received.")
        continue

    index, prob = classifier(audio_chunk)

    if index is not None and prob >= 0.6:
        print(f'Predicted Class - {labels[index]}, Probability - {prob:.2f}')
    else:
        print("Not enough data for prediction yet...")
