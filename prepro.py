import gensim
import random
from collections import Counter
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import logging
import time
from utils import VectorInit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_init = VectorInit.GLOVE
logger.info("[Vectors init mode] {}".format(vector_init.name))

max_length = 42
emb_dim = 300
version = 5
train_file = 'snli/snli_1.0_train.txt.parsed'
dev_file = 'snli/snli_1.0_dev.txt.parsed'
test_file = 'snli/snli_1.0_test.txt.parsed'

if vector_init == VectorInit.RANDOM:
    word_emb_mat_file = 'snli/word_emb_mat_random_{}'.format(version)
    tarin_record_file = 'snli/train_record_random_{}'.format(version)
    dev_record_file = 'snli/dev_record_random_{}'.format(version)
    dev_record_file = 'snli/test_record_random_{}'.format(version)
else:
    word_emb_mat_file = 'snli/word_emb_mat_{}'.format(version)
    tarin_record_file = 'snli/train_record_{}'.format(version)
    dev_record_file = 'snli/dev_record_{}'.format(version)
    test_record_file = 'snli/test_record_{}'.format(version)


def run():
    word_counter = Counter()

    # process file
    train_examples = process_file(train_file, word_counter)
    dev_examples = process_file(dev_file, word_counter)
    test_examples = process_file(test_file, word_counter)

    word_emb_mat, word2idx_dict = get_embeding(word_counter)
    np.save(word_emb_mat_file, word_emb_mat)

    build_features(train_examples, tarin_record_file, word2idx_dict)
    build_features(dev_examples, dev_record_file, word2idx_dict)
    build_features(test_examples, test_record_file, word2idx_dict)


DELIMETERS = [',', '.', '?', '!', '"', '/', ';']


def _word_tokens(sentence):
    tokens = sentence.split(' ')
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    rst = []
    for token in tokens:
        token = token.strip()
        # fix: some str end with multi ,
        while len(token) > 1:
            if token[-1] in DELIMETERS:
                token = token[:-1]
            elif token[0] in DELIMETERS:
                token = token[1:]
            else:
                break
        rst.append(token)
    return rst


def process_file(file_path, word_counter):
    examples = []
    with open(file_path, 'r') as rf:
        lines = rf.readlines()
        total = 0
        for line in tqdm(lines):
            if not line:
                continue
            items = line.split('\t')
            label_str = items[0]

            if label_str == 'entailment':
                label = [1, 0, 0]
            elif label_str == 'contradiction':
                label = [0, 1, 0]
            elif label_str == 'neutral':
                label = [0, 0, 1]
            else:
                # TODO more log for label '-'
                continue

            sentence_1 = items[1].strip()
            sentence_2 = items[2].strip()

            tokens_1 = _word_tokens(sentence_1)
            tokens_2 = _word_tokens(sentence_2)

            for token in tokens_1:
                word_counter[token] += 1
            for token in tokens_2:
                word_counter[token] += 1

            example = {
                'q1': tokens_1,
                'q2': tokens_2,
                'label': label,
                'id': total
            }
            total += 1
            examples.append(example)
        random.shuffle(examples)
        return examples


def get_embeding(word_counter):
    word_emb_mat = []
    word2idx_dict = {}

    NULL = "--NULL--"
    OOV = "--OOV--"
    word2idx_dict[NULL] = 0
    word2idx_dict[OOV] = 1

    word_emb_mat.append([0. for _ in range(emb_dim)])
    word_emb_mat.append([0. for _ in range(emb_dim)])

    embeddings_dict = None
    if vector_init == VectorInit.W2V:
        embding_file = 'GoogleNews-vectors-negative300.bin.gz'
        logger.info('Load w2v begin')
        start = time.time()

        embeddings_dict = gensim.models.KeyedVectors.load_word2vec_format(embding_file, binary=True)
        logger.info('Load w2v cost {:.2f}'.format(time.time() - start))
    elif vector_init == VectorInit.GLOVE:
        embding_file = 'glove.840B.300d.txt'
        logger.info('Load glove begin')
        start = time.time()

        embeddings_dict = {}
        f = open(embding_file, encoding='utf8')
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = coefs
        f.close()
        logger.info('Load glove cost {:.2f}'.format(time.time() - start))

    index = 2
    total = 0
    success = 0

    for word in word_counter.keys():
        total += 1
        try:
            if vector_init == VectorInit.RANDOM:
                vector = np.random.rand(emb_dim)
            else:
                vector = embeddings_dict[word]
            success += 1
            word_emb_mat.append(vector)
            word2idx_dict[word] = index
            index += 1
        except:
            # logger.exception('')
            pass
    logger.info('Rate {:.2f} {} {}'.format(float(success) / total, success, total))

    word_emb_mat_np = np.array(word_emb_mat)
    return word_emb_mat_np, word2idx_dict


def build_features(examples, record_file, word2idx_dict):
    def _padding(q_idx):
        size = len(q_idx)
        assert size <= max_length
        if size < max_length:
            return q_idx + [0] * (max_length - size)
        return q_idx

    #remove oov token (eg: stop words in w2v)
    def _token2idx(tokens):
        rst = []
        for token in tokens:
            try:
                vector =  word2idx_dict[token]
                rst.append(vector)
            except:
                pass
        return rst


    writer = tf.python_io.TFRecordWriter(record_file)
    logger.info('Saveing file {}'.format(record_file))
    for example in tqdm(examples):
        _id = example['id']

        # q1_idx = [word2idx_dict[token] for token in example['q1']]
        # q2_idx = [word2idx_dict[token] for token in example['q2']]
        q1_idx = _token2idx(example['q1'])
        q2_idx = _token2idx(example['q2'])

        q1_idx = _padding(q1_idx)
        q2_idx = _padding(q2_idx)

        label = example['label']

        record = tf.train.Example(features=tf.train.Features(feature={
            'q1_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=q1_idx)),
            'q2_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=q2_idx)),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
            'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[_id])),
        }))
        writer.write(record.SerializeToString())
    writer.close()


if __name__ == '__main__':
    run()
