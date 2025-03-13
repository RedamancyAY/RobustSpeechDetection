# RobustSpeechDetection

This repository contains the code for the paper ["Robust AI-Synthesized Speech Detection Using Feature Decomposition Learning and Synthesizer Feature Augmentation"](https://ieeexplore.ieee.org/abstract/document/10806877).


Note, we don't give the abbreviation of our method in the Paper. In your manuscript, you can use `FD-ADD` (**F**eature **D**ecomposition based **A**udio **D**eepfake **D**etection) to represent the method in our paper.

## Dataset Splits

We create custom splits for the WaveFake and LibriSeVoc datasets as they do not provide publicly available splits. Specifically, we randomly split each dataset based on unique IDs to ensure class balance for various synthesizer methods in the training, validation, and testing splits. The split ratios vary depending on the evaluation task.

We put the train/val/test splits of the WaveFake, LibriseVoc, DECRO and ASVspoof 2021 datasets in the `splits` folder for the inner evaluation task.


## Model Implementation

We put the implementation codes of our method in the `model` folder. The code is organized into different subdirectories.


### Prepare Model Environment

You have to install the required Python packages using pip:

```bash
pip install torch torchaudio torchvision librosa einops transformers
pip install torch-yin
```
where, you can use the python version 3.9 or higher (My tests are using python 3.9).

Besides, you need to install the `ffmpeg` in your system if you use the compression transformation in the training steps. Note, Torchaudio may need the version of `ffmpeg` to be lower than 7, and install ffmpeg first and then install torchaudio.

### Model Usage

you can see model usage details in the `demo.ipynb` file.


# Note

We use `ResNet` for the feature extraction in our method. But, it can also use the `transfomer` as the backbone. You can choose either one based on your preference.

# Acknowledgments

Please cite the following paper if you use our method:
```bibtex
@article{zhangRobustAISynthesizedSpeech2025,
  title = {Robust {{AI-Synthesized Speech Detection Using Feature Decomposition Learning}} and {{Synthesizer Feature Augmentation}}},
  author = {Zhang, Kuiyuan and Hua, Zhongyun and Zhang, Yushu and Guo, Yifang and Xiang, Tao},
  date = {2025},
  journaltitle = {IEEE Transactions on Information Forensics and Security},
  volume = {20},
  pages = {871--885},
  issn = {1556-6021},
  doi = {10.1109/TIFS.2024.3520001},
  url = {https://ieeexplore.ieee.org/abstract/document/10806877},
}
```
