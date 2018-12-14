import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from model import Model

from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_util import *
from beam_search import *
from rouge import Rouge
import argparse

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class Evaluate(object):
    def __init__(self, data_path, opt, batch_size = config.batch_size):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='eval',
                               batch_size=batch_size, single_pass=True)
        self.opt = opt
        time.sleep(5)

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        checkpoint = T.load(os.path.join(config.save_model_path, self.opt.load_model))
        self.model.load_state_dict(checkpoint["model_dict"])


    def valid_one_batch(self, batch):
        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(batch)
        dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(batch)
        enc_batch = self.model.embeds(enc_batch)
        enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)
        prev_s = None
        sum_temporal_srcs = None
        # -------------------------------Summarization-----------------------
        step_losses = []
        s_t = (enc_hidden[0], enc_hidden[1])

        for t in range(min(max_dec_len, config.max_dec_steps)):
            x_t = dec_batch[:, t]
            x_t = self.model.embeds(x_t)
            final_dist, s_t, ct_e, sum_temporal_srcs, prev_s = self.model.decoder(x_t, s_t, enc_out, enc_padding_mask,
                                                              ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s)
            target = target_batch[:, t]
            log_probs = T.log(final_dist + config.eps)
            step_loss = F.nll_loss(log_probs, target, reduction="none", ignore_index=self.vocab.word2id(data.PAD_TOKEN))
            step_losses.append(step_loss)

        losses = T.sum(T.stack(step_losses, 1), 1)
        batch_avg_loss = losses / dec_lens
        mle_loss = T.mean(batch_avg_loss)

        return mle_loss.item()


    def validIters(self):
        self.setup_valid()
        count = mle_total = 0
        batch = self.batcher.next_batch()
        while batch is not None:
            with T.autograd.no_grad():
                mle_loss = self.valid_one_batch(batch)
            mle_total += mle_loss
            count += 1
            batch = self.batcher.next_batch()

        mle_avg = mle_total/count

        load_file = self.opt.load_model
        print(load_file, "mle_loss:", "%.3f"%mle_avg)

    def print_original_predicted(self, decoded_sents, ref_sents, article_sents, loadfile):
        filename = "test_"+loadfile.split(".")[0]+".txt"
    
        with open(os.path.join("data",filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: "+article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def test_batch(self, print_sents = False):

        self.setup_valid()
        batch = self.batcher.next_batch()
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        while batch is not None:
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(batch)

            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

            #-----------------------Summarization----------------------------------------------------
            with T.autograd.no_grad():
                pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, self.model, start_id, end_id, unk_id)

            for i in range(len(pred_ids)):
                decoded_words = data.outputids2words(pred_ids[i], self.vocab, batch.art_oovs[i])
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)
                decoded_sents.append(decoded_words)
                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)

            batch = self.batcher.next_batch()

        load_file = self.opt.load_model

        if print_sents:
            self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)

        scores = rouge.get_scores(decoded_sents, ref_sents, avg = True)
        if self.opt.task == "get_full_scores":
            print(load_file, "scores:", scores)
        else:
            rouge_l = scores["rouge-l"]["f"]
            print(load_file, "rouge_l:", "%.4f" % rouge_l)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="validate", choices=["validate","test","get_full_scores"])
    parser.add_argument("--start_from", type=str, default="0020000.tar")
    parser.add_argument("--load_model", type=str, default=None)
    opt = parser.parse_args()

    if opt.task == "validate" or opt.task == "test":
        saved_models = os.listdir(config.save_model_path)
        saved_models.sort()
        file_idx = saved_models.index(opt.start_from)
        saved_models = saved_models[file_idx:]
        if opt.task == "validate":
            for f in saved_models:
                opt.load_model = f
                valid_processor = Evaluate(config.valid_data_path, opt)
                valid_processor.validIters()
        else:   #test
            for f in saved_models:
                opt.load_model = f
                valid_processor = Evaluate(config.test_data_path, opt)
                valid_processor.test_batch()
    else:   #get_full_scores
        valid_processor = Evaluate(config.test_data_path, opt)
        valid_processor.test_batch()
