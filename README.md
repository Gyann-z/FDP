# FDP
This is the implementation of the paper "Focus, Distinguish, and Prompt: Unleashing CLIP for Efficient and Flexible Scene Text Retrieval".

## Usage

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies.

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
$ pip install requirements.txt
```

## Datasets
The MLT dataset can be download from [here](https://drive.google.com/drive/folders/15tHbCHgw3sKHw6K6wq2PhssYJ75TVt05?usp=drive_link)

The PSTR (Phrase-level Scene Text Retrieval) dataset can be download from [here](https://drive.google.com/drive/folders/1g3YwLpDSBzodmV75YL0xt9xCu2D1gqMI?usp=sharing)
