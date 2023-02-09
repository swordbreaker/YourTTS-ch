import os
from glob import glob
from sys import platform


from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import json

PretrainedModelPath = ""
if platform == "linux" or platform == "linux2":
    PretrainedModelPath = "/scicore/home/graber0001/perity98/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/"
else:
    PretrainedModelPath = "C:/Users/tobia/AppData/Local/tts/tts_models--multilingual--multi-dataset--your_tts/"


def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0])
            text = cols[1]
            speaker_name = wav_file.split("_")[1]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items

output_path = os.path.dirname(os.path.abspath(__file__))

# dataset config for one of the pre-defined datasets
dataset_config = [BaseDatasetConfig(language="ch_DE", path="data/swissDial/", meta_file_train="metadata.txt")]

audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

vitsArgs = VitsArgs(
    use_language_embedding=True,
    embedded_language_dim=4,
    use_speaker_embedding=True,
    use_sdp=False,
)

with open(f"{PretrainedModelPath}config.json") as f:
    json_config = json.load(f)

config = VitsConfig(**json_config)

config.run_name = "swissDial_proto"
config.phoneme_language = "ch_DE"
config.output_path = output_path
config.phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
config.model_args = vitsArgs

# config = VitsConfig(
#     model_args=vitsArgs,
#     audio=audio_config,
#     run_name="swissDial_proto",
#     use_speaker_embedding=True,
#     batch_size=32,
#     eval_batch_size=16,
#     batch_group_size=0,
#     num_loader_workers=4,
#     num_eval_loader_workers=4,
#     run_eval=True,
#     test_delay_epochs=-1,
#     epochs=1000,
#     text_cleaner="multilingual_cleaners",
#     use_phonemes=False,
#     phoneme_language="ch-de",
#     phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
#     compute_input_seq_cache=True,
#     print_step=25,
#     use_language_weighted_sampler=True,
#     print_eval=False,
#     mixed_precision=False,
#     min_audio_len=32 * 256 * 4,
#     max_audio_len=160000,
#     output_path=output_path,
#     datasets=dataset_config,
#     characters=CharactersConfig(
#         characters_class="TTS.tts.models.vits.VitsCharacters",
#         pad="<PAD>",
#         eos="<EOS>",
#         bos="<BOS>",
#         blank="<BLNK>",
#         characters="!¡'(),-.:;¿?abcdefghijklmnopqrstuvwxyzµßàáâäåæçèéêëìíîïñòóôöùúûüąćęłńœśşźżƒабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїґӧ «°±µ»$%&‘’‚“`”„",
#         punctuations="!¡'(),-.:;¿? ",
#         phonemes=None,
#     ),
#     test_sentences=[
#         [
#             "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
#             "mary_ann",
#             None,
#             "en_US",
#         ],
#         [
#             "Il m'a fallu beaucoup de temps pour d\u00e9velopper une voix, et maintenant que je l'ai, je ne vais pas me taire.",
#             "ezwa",
#             None,
#             "fr_FR",
#         ],
#         ["Ich finde, dieses Startup ist wirklich unglaublich.", "eva_k", None, "de_DE"],
#         ["Я думаю, что этот стартап действительно удивительный.", "oblomov", None, "ru_RU"],
#         ["Das isch e tescht.", "gr", None, "ch_DE"],
#     ],
# )

# force the convertion of the custom characters to a config attribute
config.from_dict(config.to_dict())

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    formatter=formatter,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.load_ids_from_file(f"{PretrainedModelPath}speakers.json")
# speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# init model
model = Vits(config, ap, tokenizer, speaker_manager, language_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()