# Warning!!!
This repository is still not complete. We will upload the complete code in the following 3 days.



# RobustSpeechDetection

This repository contains the code for the paper ["Robust AI-Synthesized Speech Detection Using Feature Decomposition Learning and Synthesizer Feature Augmentation"](https://ieeexplore.ieee.org/abstract/document/10806877).




## Dataset Splits

We create custom splits for the WaveFake and LibriSeVoc datasets as they do not provide publicly available splits. Specifically, we randomly split each dataset based on unique IDs to ensure class balance for various synthesizer methods in the training, validation, and testing splits. The split ratios vary depending on the evaluation task.

We put the train/val/test splits of the WaveFake, LibriseVoc, DECRO and ASVspoof 2021 datasets in the `splits` folder for the inner evaluation task.


## Model Implementation

We put the implementation codes of our method in the `model` folder. The code is organized into different subdirectories.


### Python Environment

You have to install the required Python packages using pip:

```bash
pip install torch torchaudio torchvision librosa einops transformers
pip install torch-yin
```
where, you can use the python version 3.9 or higher (My tests are using python 3.9).






