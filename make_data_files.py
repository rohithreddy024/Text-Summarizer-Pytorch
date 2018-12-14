import os
import shutil
import collections
import tqdm
from tensorflow.core.example import example_pb2
import struct
import random
import shutil

finished_path = "data/finished"
unfinished_path = "data/unfinished"
chunk_path = "data/chunked"

vocab_path = "data/vocab"
VOCAB_SIZE = 200000

CHUNK_SIZE = 15000 # num examples per chunk, for the chunked data
train_bin_path = os.path.join(finished_path, "train.bin")
valid_bin_path = os.path.join(finished_path, "valid.bin")

def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def write_to_bin(article_path, title_path, out_file, vocab_counter = None):

    with open(out_file, 'wb') as writer:

        article_itr = open(article_path, 'r')
        titles_itr = open(title_path, 'r')
        for article in tqdm.tqdm(article_itr):
            article = article.strip()
            abstract = next(titles_itr).strip()

            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            if vocab_counter is not None:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                # abs_tokens = [t for t in abs_tokens if
                #               t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    if vocab_counter is not None:
        with open(vocab_path, 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')

def creating_finished_data():
    make_folder(finished_path)

    vocab_counter = collections.Counter()

    write_to_bin(os.path.join(unfinished_path, "train.article.txt"), os.path.join(unfinished_path, "train.title.txt"), train_bin_path, vocab_counter)
    write_to_bin(os.path.join(unfinished_path, "valid.article.filter.txt"), os.path.join(unfinished_path, "valid.title.filter.txt"), valid_bin_path)


def chunk_file(set_name, chunks_dir, bin_file):
    make_folder(chunks_dir)
    reader = open(bin_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%04d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


if __name__ == "__main__":
    delete_folder(finished_path)
    creating_finished_data()        #create bin files

    delete_folder(chunk_path)
    chunk_file("train", os.path.join(chunk_path, "train"), train_bin_path)
    chunk_file("valid", os.path.join(chunk_path, "valid"), valid_bin_path)

    #Validation data contains around 1.9 lakh examples. Create test data by borrowing 15k examples from validation data
    make_folder(os.path.join(chunk_path, "test"))
    valid_bin_chunks = os.listdir(os.path.join(chunk_path, "valid"))
    test_chunk = random.choice(valid_bin_chunks)
    shutil.move(os.path.join(chunk_path, "valid", test_chunk), os.path.join(chunk_path, "test", "test_00.bin"))




