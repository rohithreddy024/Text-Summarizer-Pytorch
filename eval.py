import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from model import Model
from torch.nn.utils import clip_grad_norm_

from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_util import *
from beam_search import *
import shutil
from rouge import Rouge
import argparse

class Validate(object):
    def __init__(self, data_path, batch_size = config.batch_size):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='eval',
                               batch_size=batch_size, single_pass=True)

        time.sleep(5)

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        checkpoint = T.load(config.load_model_path)
        self.model.load_state_dict(checkpoint["model_dict"])


    def validate_batch(self):

        self.setup_valid()
        batch = self.batcher.next_batch()
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        decoded_sents = []
        original_sents = []
        rouge = Rouge()
        while batch is not None:
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, c_t_1 = get_enc_data(batch)

            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

            with T.autograd.no_grad():
                pred_ids = beam_search_on_batch(enc_hidden, enc_out, enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab, self.model, start_id, end_id, unk_id)

            for i in range(len(pred_ids)):
                decoded_words = data.outputids2words(pred_ids[i], self.vocab, batch.art_oovs[i])
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)
                decoded_sents.append(decoded_words)
                tar = batch.original_abstracts[i]
                original_sents.append(tar)

            batch = self.batcher.next_batch()

        load_file = config.load_model_path.split("/")[-1]

        scores = rouge.get_scores(decoded_sents, original_sents, avg = True)
        rouge_l = scores["rouge-l"]["f"]
        print(load_file, "rouge_l:", "%.4f" % rouge_l)



if __name__ == "__main__":

    saved_models = os.listdir(config.save_model_path)
    saved_models.sort()

    for f in saved_models:
        config.load_model_path = os.path.join(config.save_model_path, f)
        valid_processor = Validate(config.decode_data_path)
        valid_processor.validate_batch()

