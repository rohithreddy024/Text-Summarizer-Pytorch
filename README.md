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
* Download train, valid and test pairs (article, title) of Gigaword dataset from [here](https://github.com/harvardnlp/sent-summary)
* Copy files ```train.article.txt```, ```train.title.txt```, ```valid.article.filter.txt```, ```valid.title.filter.txt```, ```input.txt``` and ```task1_ref0.txt``` to ```data/unfinished``` folder
* Files are already preprcessed

## Creating ```.bin``` files and vocab file
* The model accepts data in the form of ```.bin``` files.
* To convert ```.txt``` file into ```.bin``` file and chunk them further, run (with Python 2):
```
python make_data_files.py
```
* You can find data in ```data/chunked``` folder and vocab file in ```data``` folder

## Training
* As suggested in [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf), first pretrain the seq-to-seq model using modified MLE (with Python 3):
```
python train.py --train_mle=yes --train_rl=no --mle_weight=1.0 --rl_weight=0.0
```
* Next, find the best saved model on validation data by running (with Python 3):
```
python eval.py
```
* After finding the best model, set ```resume_training``` variable to ```True```, modify ```load_model_path``` variable in ```data_util/config.py``` and run (with Python 3):
```
python train.py --train_mle=yes --train_rl=yes --mle_weight=0.05 --rl_weight=0.95
```
for MLE + RL training (or)
```
python train.py --train_mle=no --train_rl=yes --mle_weight=0.0 --rl_weight=1.0
```
for RL training

## Validation
* Perform validation by running (with Python 3):
```
python eval.py
```
## References
* [pytorch implementation of "Get To The Point: Summarization with Pointer-Generator Networks"](https://github.com/atulkum/pointer_summarizer)
