# FDP
This is the implementation of the ACM MM 2024 paper "[Focus, Distinguish, and Prompt: Unleashing CLIP for Efficient and Flexible Scene Text Retrieval](https://dl.acm.org/doi/abs/10.1145/3664647.3680877)".

## Usage

### Runtime Environment
First, install [PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, and install the [CLIP](https://github.com/openai/CLIP) as a Python package.

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
Then, install small additional dependencies.
```bash
$ pip install -r requirements.txt
```

### Datasets
The MLT dataset can be download from [here](https://drive.google.com/drive/folders/15tHbCHgw3sKHw6K6wq2PhssYJ75TVt05?usp=drive_link)

The PSTR (Phrase-level Scene Text Retrieval) dataset can be download from [here](https://drive.google.com/drive/folders/1g3YwLpDSBzodmV75YL0xt9xCu2D1gqMI?usp=sharing)

### Reformulated CLIP Models
The reformulated CLIP models can be download from [here](https://drive.google.com/drive/folders/1sgfP85OxktzpjlEue0B4K97fsIBY18ja?usp=sharing)

### Train
```bash
$ python main.py
```

### Test
```bash
$ python test.py
```

## Citation
@inproceedings{zeng2024focus,
  title={Focus, Distinguish, and Prompt: Unleashing CLIP for Efficient and Flexible Scene Text Retrieval},
  author={Zeng, Gangyan and Zhang, Yuan and Wei, Jin and Yang, Dongbao and Zhang, Peng and Gao, Yiwen and Qin, Xugong and Zhou, Yu},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={2525--2534},
  year={2024}
}
