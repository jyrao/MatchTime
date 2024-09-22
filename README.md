# [EMNLP 2024] MatchTime: Towards Automatic Soccer Game Commentary Generation (EMNLP 2024)
This repository contains the official PyTorch implementation of MatchTime: https://arxiv.org/abs/2406.18530/

<div align="center">
   <img src="./assets/teaser.png">
</div>

<div align="center">
   <img src="./assets/commentary.png">
</div>

## Some Information
[Project Page](https://haoningwu3639.github.io/MatchTime/)  $\cdot$ [Paper](https://arxiv.org/abs/2406.18530/) $\cdot$ [Dataset](https://drive.google.com/drive/folders/14tb6lV2nlTxn3VygwAPdmtKm7v0Ss8wG) $\cdot$ [Checkpoint](https://huggingface.co/Homie0609/MatchVoice) $\cdot$ [Demo Video (YouTube)](https://www.youtube.com/watch?v=E3RxHR-M6y0) $\cdot$ [Demo Video (bilibili)](https://www.bilibili.com/video/BV1L4421U76m)

## Requirements
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0](https://pytorch.org/) (If use A100)
- transformers >= 4.42.3
- pycocoevalcap >= 1.2

A suitable [conda](https://conda.io/) environment named `matchtime` can be created and activated with:
```
cd MatchTime
conda env create -f environment.yaml
conda activate matchtime
```

## Training
Before training, make sure you have prepared [features](https://pypi.org/project/SoccerNet/) and caption [data]((https://drive.google.com/drive/folders/14tb6lV2nlTxn3VygwAPdmtKm7v0Ss8wG)), and put them into according folders. The structure after collating should be like:
``````
└─ MatchTime
    ├─ dataset
    │     ├─ MatchTime
    │     │   ├─ valid
    │     │   └─ train
    │     │       ├─ england_epl_2014-2015
    │     │      ...     ├─ 2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
    │     │             ...    └─ Labels-caption.json
    │     │
    │     ├─ SN-Caption
    │     └─ SN-Caption-test-align
    │         ├─ england_epl_2015-2016 
    │        ...  ├─ 2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea
    │            ...           └─ Labels-caption_with_gt.json
    │
    ├─ features
    │     ├─ baidu_soccer_embeddings
    │     │   ├─ england_epl_2014-2015 
   ...    │  ...  ├─ 2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
          │      ...     ├─ 1_baidu_soccer_embeddings.npy
          │              └─ 2_baidu_soccer_embeddings.npy
          ├─ C3D_PCA512
         ...
``````
with the format of features is adjusted by
```
python ./features/preprocess.py directory_path_of_feature
```
Above example gives the format of Baidu feature, in our experiments we also used ResNET_PCA_512, C3D_PCA_512 from official website. If you want to use [CLIP](https://github.com/openai/CLIP)(2 FPS) or [InternVideo](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo1)(1FPS) feature. You can follow their official website to extract feature or contact us for features.

After preparing the data and features, you can pre-train (or finetune) with the following terminal command (Check hyper-parameters at the bottom of *train.py*):
```
python train.py
```
## Inference

We provide two types of inference:

#### For all test set

You can generate a *.csv* file with the following code to test the ***MatchVoice*** model with the following code (Check hyper-parameters at the bottom of *inference.py*)

```
python inference.py
```

There is a sample of this type of inference in *./inference_result/sample.csv*.

#### For Single Video

We also provide a version for predict the commentary single video (for our checkpoints, use 30s video)
```
python inference_single_video_CLIP.py single_video_path
```
Here we only provide the version of CLIP feature (using VIT/B-32), for crop the CLIP feature, please check [here](https://github.com/openai/CLIP). CLIP features are not the one with best performance but are the most friendly for new new videos.

## Alignment

Before doing alignment, you should download videos from [here](https://www.soccer-net.org/data) (224p is enough) and make it in the following format:

``````
└─ MatchTime
    ├─ videos_224p
   ...    ├─ england_epl_2014-2015
         ...   ├─ 2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
              ...     ├─ 1_224.mkv
                      └─ 2_224p.mkv
``````

### Pre-process (Coarse Align)

We need to use [WhisperX](https://github.com/m-bain/whisperX) and [LLaMA3](https://huggingface.co/docs/transformers/model_doc/llama3) (as agent) to finish coarse alignment with following steps:

*WhisperX ASR:*
```
python ./alignment/soccer_whisperx.py --process_directory video_folder(eg. ./videos_224p/england_epl_2014-2015) --output_directory output_folder(eg. ./ASR_results/england_epl_2014-2015)
``` 
*Transform to Events:*
```
python ./alignment/soccer_asr2events.py --base_path ASR_results_folder(eg. ./ASR_results/england_epl_2014-2015) --output_dir envent_results_folder(eg. ./event_results/england_epl_2014-2015)
```

*Align from Events:*
```
python ./alignment/soccer_align_from_event.py --event_path envent_results_folder(eg. ./event_results/england_epl_2014-2015) --output_dir output_directory(eg. ./pre-processed/england_epl_2014-2015)
```

More details could be checked in paper.

### Contrastive Learning (Fine-grained Align)

After downloading checkpoints from [here](https://huggingface.co/Homie0609/MatchTime/tree/main). Use the following code to finish alignment with contrastive learning:
```
python ./alignment/do_alignment.py
```
By changing the hyper-parameter ***finding_words***, you can freely align from ASR, enent, or original SN-Caption.

Also, you can directly use alignment model by
```
from alignment.matchtime_model import ContrastiveLearningModel
```

## Evaluation
We provide codes for evaluate the prediction results:
```
# for single csv file
python ./evaluation/scoer_single.py --csv_path ./inference_result/sample.csv
# for many csv files to record scores in a new csv file
python ./evaluation/scoer_group.py
# for gpt score (need OpenAI API Key)
python ./evaluation/scoer_gpt.py ./inference_result/sample.csv
```

## TODO
- [x] Commentary Model & Training & Inference Code
- [x] Release Checkpoints
- [x] Release Meta Data
- [x] Alignment Model & Training & Inference Code
- [x] Evaluation Code
- [x] Release Demo

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
