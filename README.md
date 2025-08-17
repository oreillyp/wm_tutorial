# Interspeech 2025 Tutorial on Speech Watermarking

This repository contains materials for the Interspeech 2025 Tutorial on Speech Watermarking, including slides, accompanying notebooks, and utility code. The goal of this tutorial is to give participants a broad introduction to research in the watermarking of speech audio, primarily as it relates to identifying synthetic or "AI-generated" recordings.

Links to code are provided below:

| __Notebook__ | __Link__ |
|--|--|
| Example 1: Basics | <a href="https://colab.research.google.com/github/oreillyp/wm_tutorial/blob/main/notebooks/basics.ipynb"><img alt="Open Example 1 in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" /></a> |
| Example 2: EigenWatermark | <a href="https://colab.research.google.com/github/oreillyp/wm_tutorial/blob/main/notebooks/eigenwatermark.ipynb"><img alt="Open Example 2 in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" /></a> |
| Example 3: End-to-End Neural Network Watermark | <a href="https://colab.research.google.com/github/oreillyp/wm_tutorial/blob/main/notebooks/end_to_end.ipynb"><img alt="Open Example 3 in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" /></a> |
| Example 4: Attacks on Speech Watermarks | <a href="https://huggingface.co/spaces/oreillyp/watermark_stress_test"><img alt="Open Example 4 in HuggingFace" src="https://img.shields.io/badge/Open%20in-HuggingFace-blue?logo=huggingface" /></a> |

## Linking Google Colab to Google Drive

If you will be running notebooks via [Google Colab](https://colab.research.google.com/), you'll want to avoid repeatedly downloading code and data in each separate notebook due to the lack of a persistent shared runtime. You can get around this by [linking your Google Drive account](https://www.marktechpost.com/2025/07/12/how-to-connect-google-colab-with-google-drive/). At the top of each notebook in this repository is a set of cells handling installation steps for Google Colab; when you run the cell containing 
```
from google.colab import drive
drive.mount('/content/drive')
```
you will be prompted with instructions in a pop-up window to grant permissions and complete the setup. When you are done, you should see a `drive` folder in the Colab file explorer (accessible via the folder icon in the left sidebar).

## Installation

If you are running a notebook in Google Colab, run the cells at the top of the notebook to install the code and necessary dependencies. If you are running a notebook on your own machine, you only need to install once via:

```
git clone https://github.com/oreillyp/wm_tutorial.git
cd  wm_tutorial && pip install -e .
```

## Data Download

We provide a script to download ~14G of speech, reverb, and noise data used for training and evaluating watermarking systems. Whether you are running code in Colab or on your own compute, this only needs to be run once.

In Colab, open the terminal and run:
```
cd /content/drive/MyDrive/wm_tutorial
chmod -R u+x scripts/
./scripts/download_data.sh /content/drive/MyDrive/wm_tutorial_data
```

If you are not using Colab, navigate to the `wm_tutorial` repository and run the following terminal commands:
```
chmod -R u+x scripts/
./scripts/download_data.sh <DATA_DIR>
```
where `<DATA_DIR>` is the path to the directory in which you wish to store all data.

In a free-tier Google Colab notebook environment, the data download should take roughly 10 minutes. We source speech data from he [LibriTTS-R](https://www.openslr.org/141/) dataset (specifically, the `train-clean-100` and `test-clean` subsets). To simulate acoustic distortions such as background noise and reverberation, we use the [RIR/Noise database](https://www.openslr.org/28/).


