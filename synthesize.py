from TTS.api import TTS

tts = TTS(
    model_path="./swissDial-February-24-2023_10+21PM-fe3680a/best_model.pth",
    config_path="./swissDial-February-24-2023_10+21PM-fe3680a/config.json",
    progress_bar=True, gpu=False)

# tts.tts_to_file("Das isch en Tescht.", file_path="output1.wav")
tts.tts_to_file("Das isch e Test.", speaker_wav="./data/slowsoft/s_11000.wav", file_path="output.wav")
# tts.tts_to_file("Das isch e Test.", speaker_wav="data/SDS-200/1adbd534178954e45abd23e1804f125bb0365eca4f0089d5e6408b8fa4b1c412.wav", file_path="output3.wav")
# tts.tts_to_file("Das isch e andere Test.", speaker_wav="data/slowsoft/s_11000_16000.wav", file_path="output4.wav")
# tts.tts_to_file("Das isch e andere Test.", speaker_wav="data/swissDial_16000/wavs/ch_ag_0000.wav", file_path="output5.wav")