# Your TTS

## Installation on scicore
```bash
ml CUDA/11.7.0
ml Miniconda2/4.3.30
conda create -n tts python=3.9
source activate tts
git clone https://github.com/coqui-ai/TTS
cd TTS
conda install -c conda-forge tts 
pip install -e .
```

## Run training
```
sbatch train_yourtts.sh
```

For more information see [https://tts.readthedocs.io/en/latest/](https://tts.readthedocs.io/en/latest/)