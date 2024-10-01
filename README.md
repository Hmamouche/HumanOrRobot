## Introduction
This repository contains the data and the code for reproducing the results of the paper:
__Listen and tell me who the user is talking to: Automatic detection of
the interlocutor’s type during a conversation__.
Only the intermediate extracted features from raw data that are available for this repository as described in the paper, they can be downloaded from https://drive.google.com/drive/folders/1vFXsOdpKmfP-mBS-ZvMuKKTdYD59MCqx?usp=sharing. To get the raw data, i.e., conversational multimodal signals of 25 subjects, please contact us.

## Training

- Using linguistic features:
```bash
python src/train_models.py -t ling -m att-lstm
```

- Using filter banks features
```bash
python src/train_models.py -t ling -m att-lstm
```
- __Remarque__: use python src/train_models.py -h to train more models.

## Testing

- Usinglinguistic features
```bash
python src/test_models.py -t ling -m att-lstm
```

- Using filter banks features
```bash
python src/test_models.py -t ling -m att-lstm
```
