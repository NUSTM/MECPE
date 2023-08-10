# Multimodal Emotion-Cause Pair Extraction in Conversations

This repository contains the dataset and code for our TAFFC 2022 paper:

F. Wang, Z. Ding, R. Xia, Z. Li and J. Yu, "[Multimodal Emotion-Cause Pair Extraction in Conversations](https://ieeexplore.ieee.org/document/9969873)," in IEEE Transactions on Affective Computing, doi: 10.1109/TAFFC.2022.3226559.

🔥 We have organized a SemEval task based on our ECF dataset, and will release the source data of the three modalities. Welcome to participate in the competition. Visit [SemEval-2024 Task 3: The Competition of Multimodal Emotion Cause Analysis in Conversations](https://nustm.github.io/SemEval-2024_ECAC/).

## Dependencies

- Python 3 (tested on python 3.6.13)
- [Tensorflow](https://github.com/tensorflow/tensorflow) 1.15.5
- [BERT](https://github.com/google-research/bert) (The pretrained BERT model "[BERT-Base, Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)" is required.)

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
