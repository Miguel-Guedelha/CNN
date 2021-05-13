import codecs
import glob
import re
import string
import xml.etree.ElementTree as et
from collections import Counter
from os import path
from tqdm import tqdm

import numpy as np
import numpy.random as random
import pandas as pd
import torch
from nltk.corpus import stopwords

random.seed(42)


def load_and_process(dir: str, type):
    dirXmls = glob.glob(dir + '*.xml')
    data = []
    idsList = []
    for file in dirXmls:
        tree = et.parse(file)
        root = tree.getroot()
        doc = root.find('documents').findall('document')
        if type == 0:
            tweets = [process_all(tweet.text) for tweet in doc]
        elif type == 1:
            tweets = [only_mentions(tweet.text) for tweet in doc]
        else:
            tweets = [only_tags(tweet.text) for tweet in doc]

        tweets = [x for x in tweets if x is not ""]
        #wrong_per_tweet.append(sum([len(spell.unknown(tweet.text.split())) for tweet in doc])/len(doc))
        ids = path.split(file)[-1][0:-4]
        data.append(tweets)
        idsList.append(ids)

    return data, idsList


def load_truth(path):
    truth = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            id = line[:-5]
            label = line[-2]
            truth[id] = label
    return truth


def append_truth(data, truth, ids):
    zipped = zip(data, ids)
    train_data_labels = []
    for d, id in zipped:
        label = int(truth[id])
        label = torch.FloatTensor([label])
        train_data_labels.append([d, label])
    return train_data_labels


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    out = emoji_pattern.sub(r'', text)

    return out


def build_vocab(data, tokenizer):
    counter = Counter()
    max_length = 0
    for tweets in data:
        tweets_joined = ' '.join(tweets)
        tokens = tokenizer(tweets_joined)
        if len(tokens) > max_length:
            max_length = len(tokens)
        counter.update(tokens)

    word2idx = {}
    word2idx["<pad>"] = 0
    word2idx["<unk>"] = 1
    i = 2
    for word in counter.keys():
        word2idx[word] = i
        i += 1

    return word2idx, max_length


def create_tensors(data, vocab, tokenizer, max_length):
    tensor_data = []
    pad = vocab["<pad>"]

    for tweets in data:
        tweets_joined = ' '.join(tweets)
        tokens = [vocab[token] for token in tokenizer(tweets_joined)]
        tokens += [pad] * (max_length - len(tokens))
        tensor = torch.tensor(tokens, dtype=torch.long)
        tensor_data.append(tensor)
    return torch.stack(tensor_data)


def create_weight_matrix(embedding, embedding_dim, vocab, seed):
    random.seed(seed)
    matrix = np.zeros((len(vocab), embedding_dim))

    for (index, word) in enumerate(vocab.itos):
        try:
            matrix[vocab[word]] = embedding[word]
        except KeyError:
            if word == "<pad>":
                matrix[vocab[word]] = [0. for _ in range(embedding_dim)]
            else:
                matrix[vocab[word]] = random.normal(
                    scale=0.25, size=(embedding_dim,))
    return torch.FloatTensor(matrix)


def remove_masks(text):
    mask_patterns = re.compile(r"#.+#")
    return mask_patterns.sub(r'', text)


def remove_stopwords(text):
    stop = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop])
    return text


def remove_punctuation(text: str):
    punctuation = string.punctuation + '\u2026' + '-'
    return text.translate(str.maketrans('', '', punctuation))


def process_all(text):
    #text = remove_emoji(text)
    text = re.sub('[^0-9a-z #@]', "", text)
    text = re.sub('[\n]', " ", text)
    #temp = remove_stopwords(temp)
    #temp = remove_masks(temp)
    #temp = remove_punctuation(temp)
    return text.lower()


def only_tags(text):
    text = re.findall(r"#\w+", text)
    if text != []:
        text = ' '.join(text)
    else:
        text = ''
    return text.lower()


def only_mentions(text):
    text = re.findall(r"@\w+", text)
    if text != []:
        text = ' '.join(text)
    else:
        text = ''
    return text.lower()


def create_frame(data, ids):
    return pd.DataFrame(data, index=ids, columns=['data', 'truth'])


def load_embedding(filename, vocab, dim):
    file = open(filename, 'r', encoding='utf8', newline='\n', errors='ignore')

    embedding = np.random.uniform(-0.25, 0.25, (len(vocab), dim))
    embedding[vocab["<pad>"]] = np.zeros((dim,))
    next(file, None)
    for line in tqdm(file):
        key_values = line.rstrip().split(' ')
        word = key_values[0]
        if word in vocab:
            embedding[vocab[word]] = np.array(
                key_values[1:], dtype=np.float32)
    return embedding


def convert_to_binary(path):
    f = codecs.open(path + '.txt', 'r', encoding='utf8')
    wv = []

    with codecs.open(path + '.vocab', 'w', encoding='utf8') as vocab_write:
        for line in f:
            line_split = line.split()
            if(len(line_split) != 0):
                vocab_write.write(line_split[0].strip())
                vocab_write.write("\n")
                wv.append([float(val) for val in line_split[1:]])
    np.save(path + ".npy", np.array(wv))


def load_binary_embedding(path):
    with codecs.open(path + '.vocab', 'r', encoding='utf8') as f_in:
        index2word = [line.strip() for line in f_in]

    wv = np.load(path + '.npy', allow_pickle=True)
    embedding = {}
    for i, w in enumerate(index2word):
        embedding[w] = wv[i]
    return embedding
