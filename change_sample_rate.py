import librosa
import soundfile as sf
import os
from tqdm import tqdm

in_path = "data/slowsoft/wavs/"
out_path = "data/slowsoft_1600/wavs/"

for file in tqdm(os.listdir("data/slowsoft/wavs/")):
    audio, sr = sf.read(in_path + file)
    resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sf.write(out_path + file, audio, 16000)
