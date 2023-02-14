from TTS.api import TTS

tts = TTS(
    model_path="./swissDial_16000-February-14-2023_10+35AM-9002191/best_model.pth",
    config_path="./swissDial_16000-February-14-2023_10+35AM-9002191/config.json",
    progress_bar=True, gpu=False)

# tts.tts_to_file("Das isch en Tescht.", speaker="ag", file_path="output.wav")
tts.tts_to_file("Das isch e Test.", speaker_wav="./data/SDS-200/trimmed.wav", file_path="output.wav")
# tts.tts_to_file("Das isch e Test.", speaker_wav="data/SDS-200/1adbd534178954e45abd23e1804f125bb0365eca4f0089d5e6408b8fa4b1c412.wav", file_path="output.wav")