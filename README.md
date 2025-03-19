# Multimodal Emotion-Cause Pair Extraction in Conversations

[![Dataset](https://img.shields.io/badge/Dataset-🤗_Hugging_Face-F0A336)](https://huggingface.co/datasets/NUSTM/ECF) [![Journal](https://img.shields.io/badge/Paper-TAFFC-2E6396)](https://ieeexplore.ieee.org/document/9969873) [![Conference](https://img.shields.io/badge/Paper-arXiv:2110.08020-A42F22)](https://arxiv.org/abs/2110.08020) [![Website](https://img.shields.io/badge/Competition-SemEval_2024-488DF8)](https://nustm.github.io/SemEval-2024_ECAC/)

This repository contains the dataset and code for our IEEE TAFFC 2023 paper: [Multimodal Emotion-Cause Pair Extraction in Conversations](https://ieeexplore.ieee.org/document/9969873)
Please [**cite**](#Citation) our paper according to the official format.

🌟 Our **task paper** for SemEval-2024 is available [here](https://aclanthology.org/2024.semeval2024-1.273).

🌟 We have organized a SemEval task based on our ECF dataset and extended it with additional evaluation data. Welcome to participate in our competition. Visit [SemEval-2024 Task 3: Multimodal Emotion Cause Analysis in Conversations](https://nustm.github.io/SemEval-2024_ECAC/). 

## Dependencies

- Python 3.6.9 (tested on cuda 10.2 and NVIDIA RTX TITAN)
- [Tensorflow](https://github.com/tensorflow/tensorflow) 1.15.4
- [BERT](https://github.com/google-research/bert) (The pretrained BERT model "[BERT-Base, Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)" is required.)

> [!NOTE]
> For your convenience, we also provide our requirements.txt.cuda11+ (tested on cuda 11.3 and python 3.6.13).

## Usage

Step1:
```
# For Task 1: MECPE
python -u step1.py --use_x_a yes  --use_x_v yes --scope BiLSTM_A_V
python -u step1.py --model_type BERTcased --use_x_a yes  --use_x_v yes --scope BERT_A_V
# For Task 2: MECPE-Cat
python -u step1.py --choose_emocate yes --use_x_a yes  --use_x_v yes --scope BiLSTM_A_V_emocate
python -u step1.py --model_type BERTcased --choose_emocate yes --use_x_a yes  --use_x_v yes --scope BERT_A_V_emocate
```

Step2:
```
python -u step2.py  --use_x_a yes  --use_x_v yes   --scope  BiLSTM_A_V
```

## <span id="Citation">Citation</span>

```
@ARTICLE{wang2023multimodal,
  author={Wang, Fanfan and Ding, Zixiang and Xia, Rui and Li, Zhaoyu and Yu, Jianfei},
  journal={IEEE Transactions on Affective Computing}, 
  title={Multimodal Emotion-Cause Pair Extraction in Conversations}, 
  year={2023},
  volume={14},
  number={3},
  pages={1832-1844},
  doi = {10.1109/TAFFC.2022.3226559}
}

@InProceedings{wang2024SemEval,
  author={Wang, Fanfan  and  Ma, Heqing  and  Xia, Rui  and  Yu, Jianfei  and  Cambria, Erik},
  title={SemEval-2024 Task 3: Multimodal Emotion Cause Analysis in Conversations},
  booktitle={Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)},
  month={June},
  year={2024},
  address={Mexico City, Mexico},
  publisher={Association for Computational Linguistics},
  pages={2022--2033},
  url = {https://aclanthology.org/2024.semeval2024-1.273}
}
```
