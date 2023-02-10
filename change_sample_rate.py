import librosa
import soundfile as sf
import os
import os.path
from tqdm import tqdm

in_path = "data/swissDial/wavs/"
out_path = "data/swissDial_1600/wavs/"

for file in tqdm(os.listdir(in_path)):
    if(os.path.isfile(out_path + file)):
        continue
    audio, sr = sf.read(in_path + file)
    resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sf.write(out_path + file, audio, 16000)
