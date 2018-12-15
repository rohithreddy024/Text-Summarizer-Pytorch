# Text-Summarizer-Pytorch
Combining [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/pdf/1705.04304.pdf) and [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)

## Model Description
* LSTM based Sequence-to-Sequence model for Abstractive Summarization
* Pointer mechanism for handling Out of Vocabulary (OOV) words [See et al. (2017)](https://arxiv.org/pdf/1704.04368.pdf)
* Intra-temporal and Intra-decoder attention for handling repeated words [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf)
* Self-critic policy gradient training along with MLE training [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf)

## Prerequisites
* Pytorch
* Tensorflow
* Python 2 & 3
* [rouge](https://github.com/pltrdy/rouge) 

## Data
* Download train and valid pairs (article, title) of OpenNMT provided Gigaword dataset from [here](https://github.com/harvardnlp/sent-summary)
* Copy files ```train.article.txt```, ```train.title.txt```, ```valid.article.filter.txt```and ```valid.title.filter.txt``` to ```data/unfinished``` folder
* Files are already preprcessed

## Creating ```.bin``` files and vocab file
* The model accepts data in the form of ```.bin``` files.
* To convert ```.txt``` file into ```.bin``` file and chunk them further, run (requires Python 2 & Tensorflow):
```
python make_data_files.py
```
* You can find data in ```data/chunked``` folder and vocab file in ```data``` folder

## Training
* As suggested in [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf), first pretrain the seq-to-seq model using MLE (with Python 3):
```
python train.py --train_mle=yes --train_rl=no --mle_weight=1.0
```
* Next, find the best saved model on validation data by running (with Python 3):
```
python eval.py --task=validate --start_from=0005000.tar
```
* After finding the best model (lets say ```0100000.tar```) with high rouge-l f score, load it and run (with Python 3):
```
python train.py --train_mle=yes --train_rl=yes --mle_weight=0.25 --load_model=0100000.tar
```
for MLE + RL training (or)
```
python train.py --train_mle=no --train_rl=yes --mle_weight=0.0 --load_model=0100000.tar
```
for RL training

## Validation
* To perform validation of RL training, run (with Python 3):
```
python eval.py --task=validate --start_from=0100000.tar
```
## Testing
* After finding the best model of RL training (lets say ```0200000.tar```), evaluate it on test data & get all rouge metrics by running (with Python 3):
```
python eval.py --task=test --load_model=0200000.tar
```

## References
* [pytorch implementation of "Get To The Point: Summarization with Pointer-Generator Networks"](https://github.com/atulkum/pointer_summarizer)
