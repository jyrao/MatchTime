# MatchTime: Towards Automatic Soccer Game Commentary Generation
This repository contains the official PyTorch implementation of MatchTime: https://arxiv.org/abs/2406.18530/

<div align="center">
   <img src="./teaser.png">
</div>

<div align="center">
   <img src="./commentary.png">
</div>

## Some Information
[Project Page](https://haoningwu3639.github.io/MatchTime/)  $\cdot$ [Paper](https://arxiv.org/abs/2406.18530/) $\cdot$ [Dataset](https://drive.google.com/drive/folders/14tb6lV2nlTxn3VygwAPdmtKm7v0Ss8wG) $\cdot$ [Checkpoint](https://huggingface.co/) (Soon)

## Requirements
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.12](https://pytorch.org/)

A suitable [conda](https://conda.io/) environment named `matchtime` can be created and activated with:
```
conda env create -f environment.yaml
conda activate matchtime
```


## Training

## Inference

## TODO
- [ ] Release Demo
- [ ] Model & Training & Inference Code
- [ ] Dataset Processing Pipeline
- [ ] Meta Data
- [ ] Release Checkpoints

## Citation
If you use this code for your research or project, please cite:

	@arxiv{rao2024matchtimeautomaticsoccergame,
      title={MatchTime: Towards Automatic Soccer Game Commentary Generation}, 
      author={Jiayuan Rao and Haoning Wu and Chang Liu and Yanfeng Wang and Weidi Xie},
      year={2024},
      journal={arXiv preprint arXiv:2406.18530},
      }

## Acknowledgements
Many thanks to the code bases from [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA) and source data from [SoccerNet-Caption](https://arxiv.org/abs/2304.04565).

## Contact
If you have any questions, please feel free to contact jy_rao@sjtu.edu.cn or haoningwu3639@gmail.com.