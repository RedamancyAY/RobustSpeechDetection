# RobustSpeechDetection


## Dataset Splits

We create custom splits for the WaveFake and LibriSeVoc datasets as they do not provide publicly available splits. Specifically, we randomly split each dataset based on unique IDs to ensure class balance for various synthesizer methods in the training, validation, and testing splits. The split ratios vary depending on the evaluation task.

We put the train/val/test splits of the WaveFake, LibriseVoc, DECRO and ASVspoof 2021 datasets in the `splits` folder for the inner evaluation task.


## Models

We put the implementation codes of the WaveLM and Wave2Vec in the `models` folder.