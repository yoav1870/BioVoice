import torch

model = torch.hub.load(
    "IDRnD/ReDimNet",
    "ReDimNet",
    model_name="b6",  # or b6
    train_type="ptn",
    dataset="vox2",
)
model.eval()

sr = 16000
seconds = 4.5
num_samples = int(sr * seconds)

wav = torch.zeros(1, num_samples)

with torch.no_grad():
    mel = model.spec(wav)

print("waveform shape:", tuple(wav.shape))
print("mel shape:", tuple(mel.shape))
print("T =", mel.shape[-1])
