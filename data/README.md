## ECF dataset 



For our task MECPE, we accordingly construct a multimodal conversational emotion cause dataset named Emotion-Cause-in-Friends (ECF) from the _Friends_ sitcom. We choose [MELD](https://github.com/declare-lab/MELD) as the data source and further annotate the corresponding causes for the given emotion annotations.

The complete dataset in JSON format is available on Hugging Face. [![Dataset](https://img.shields.io/badge/🤗-ECF-F0A336)](https://huggingface.co/datasets/NUSTM/ECF)

### Dataset Statistics

| Item          | Train | Dev | Test | Total |
| ------------- | ----- | --- | ---- | ----- |
| Conversations | 1001   | 112 | 261  | 1,374   |
| Utterances    | 9,966   | 1,087 | 2,566  | 13,619   |
| Emotion (utterances)    | 5,577   | 668 | 1,445  | 7,690   |
| Emotion-cause (utterance) pairs    | 7,055   | 866 | 1,873  | 9,794   |

### File Description

The conversation data for the training, validation, and test sets are stored separately in the files `train.txt` , `dev.txt` and `test.txt`, with the complete dataset available in `all_data_pair.txt`. The data format is as follows:

```
5 3  ## Conversation_ID Utterance_ID  
(1,3),(3,3)  ## Emotion_Cause_Utterance_Pairs 
1 | Rachel | joy | Oh , look , wish me luck ! | Friends_S1E1: 00:16:45.504 - 00:16:46.672  ## Utterance_No. | Speaker_Name | Emotion | Utterance_Text | Timestamps_in_Friends

...
```

The file  `all_data_pair_ECFvsMELD.txt` contains the correspondence between utterances in ECF and those in MELD.
For example, the line `4 | Joey | surprise | Instead of ... ? | train_dia559_utt3` means that this utterance corresponds to utterance 3 of the dialogue 559 from the MELD training set, allowing you to match it with MELD's video clips.

### About Multimodal Data   

⚠️ Due to potential copyright issues, we do not provide pre-segmented video clips. 

If you need to utilize multimodal data, you may consider the following options:

1. Use the acoustic and visual features we provide:
    - [`audio_embedding_6373.npy`](https://drive.google.com/file/d/1EhU2jFSr_Vi67Wdu1ARJozrTJtgiQrQI/view?usp=share_link): the embedding table composed of the 6373-dimensional acoustic features of each utterances extracted with openSMILE
    - [`video_embedding_4096.npy`](https://drive.google.com/file/d/1NGSsiQYDTqgen_g9qndSuha29JA60x14/view?usp=share_link): the embedding table composed of the 4096-dimensional visual features of each utterances extracted with 3D-CNN

2. Since ECF is constructed based on the MELD dataset, you can download the raw video clips from [MELD](https://github.com/declare-lab/MELD). 
Most utterances in ECF align with MELD. However, **we have made certain modifications to MELD's raw data while constructing ECF, including but not limited to editing utterance text, adjusting timestamps, and adding or removing utterances**. Therefore, some timestamps provided in ECF have been corrected, and there are also new utterances that cannot be found in MELD. Given this, we recommend option (3) if feasible.

3. Download the raw videos of _Friends_ from the website, and use the FFmpeg toolkit to extract audio-visual clips of each utterance based on the timestamps we provide.





  

