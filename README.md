
Data totals 14G.

In a free-tier Google Colab notebook environment, the data download should take roughly 10 minutes.

## Data

We will train and evaluate on speech from the [LibriTTS-R]() dataset (respectively, the `train-clean-100` and `test-clean` subsets). To simulate acoustic distortions such as background noise and reverberation, we will use the [RIR/Noise database]().

While we will train our watermarking systems at a 16kHz sample rate, we will demonstrate how to easily adapt them to higher sample rates via band-splitting.