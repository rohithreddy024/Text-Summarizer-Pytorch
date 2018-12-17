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
* You will find the data in ```data/chunked``` folder and vocab file in ```data``` folder

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
python train.py --train_mle=yes --train_rl=yes --mle_weight=0.25 --load_model=0100000.tar --new_lr=0.0001
```
for MLE + RL training (or)
```
python train.py --train_mle=no --train_rl=yes --mle_weight=0.0 --load_model=0100000.tar --new_lr=0.0001
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

## Results
* Rouge scores obtained by using best MLE trained model on test set:  
scores: {  
```'rouge-1':``` {'f': 0.4412018559893622, 'p': 0.4814799494024485, 'r': 0.4232331027817015},  
```'rouge-2':``` {'f': 0.23238981595683728, 'p': 0.2531296070596062, 'r': 0.22407861554997008},  
```'rouge-l':``` {'f': 0.40477682528278364, 'p': 0.4584684491434479, 'r': 0.40351107200202596}  
}

* Rouge scores obtained by using best MLE + RL trained model on test set:  
scores: {  
```'rouge-1':``` {'f': 0.4499047033247696, 'p': 0.4853756369556345, 'r': 0.43544461386607497},  
```'rouge-2':``` {'f': 0.24037014314625643, 'p': 0.25903387205387235, 'r': 0.23362662645146298},  
```'rouge-l':``` {'f': 0.41320241732946406, 'p': 0.4616655167980162, 'r': 0.4144419466382236}  
}

* Training log file is included in the repository

# Examples
```article:``` russia 's lower house of parliament was scheduled friday to debate an appeal to the prime minister that challenged the right of u.s.-funded radio liberty to operate in russia following its introduction of broadcasts targeting chechnya .  
```ref:``` russia 's lower house of parliament mulls challenge to radio liberty  
```dec:``` russian parliament to debate on banning radio liberty  

```article:``` continued dialogue with the democratic people 's republic of korea is important although australia 's plan to open its embassy in pyongyang has been shelved because of the crisis over the dprk 's nuclear weapons program , australian foreign minister alexander downer said on friday .  
```ref:``` dialogue with dprk important says australian foreign minister  
```dec:``` australian fm says dialogue with dprk important  

```article:``` water levels in the zambezi river are rising due to heavy rains in its catchment area , prompting zimbabwe 's civil protection unit -lrb- cpu -rrb- to issue a flood alert for people living in the zambezi valley , the herald reported on friday .  
```ref:``` floods loom in zambezi valley  
```dec:``` water levels rising in zambezi river  

```article:``` tens of thousands of people have fled samarra , about ## miles north of baghdad , in recent weeks , expecting a showdown between u.s. troops and heavily armed groups within the city , according to u.s. and iraqi sources .  
```ref:``` thousands flee samarra fearing battle  
```dec:``` tens of thousands flee samarra expecting showdown with u.s. troops  

```article:``` the #### tung blossom festival will kick off saturday with a fun-filled ceremony at the west lake resort in the northern taiwan county of miaoli , a hakka stronghold , the council of hakka affairs -lrb- cha -rrb- announced tuesday .  
```ref:``` #### tung blossom festival to kick off saturday  
```dec:``` #### tung blossom festival to kick off in miaoli  

## References
* [pytorch implementation of "Get To The Point: Summarization with Pointer-Generator Networks"](https://github.com/atulkum/pointer_summarizer)
