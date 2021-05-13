import random
import utility
import cnn
import torch
from torchtext.data.utils import get_tokenizer

embedding_path = f'embeddings/crawl-300d-2M.vec'

#Change folder after "data/" according to dataset to test
#   fake-vs-real-unfiltered/  -> fake news spreaders vs real with tags and mentions unfiltered  
#   fake-vs-non_fake-unfiltered/  -> fake news spreaders vs non fake news spreaders with tags and mentions unfiltered   
#base = 'data/fake-vs-real-unfiltered/'
#base = 'data/fake-vs-non_fake-unfiltered/'
base = 'data/pan20/'
test_dir = base + 'test/'
train_dir = base + 'train/'
dim = 300

#Type of testing to be made:
#   n-grams = 0
#   user mentions = 1
#   hashtags = 2
type = 0

#Path to model that we want to test, depending on type
model_path = 'models/covid_4epochs_95.0_accuracy_v2.pth'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_data, train_ids = utility.load_and_process(train_dir, type)
ground_truth_train = utility.load_truth(train_dir + 'truth.txt')

test_data, test_ids = utility.load_and_process(test_dir, type)
ground_truth_test = utility.load_truth(test_dir + 'truth.txt')

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

vocab_data = train_data + test_data

word2idx, max_length = utility.build_vocab(vocab_data, tokenizer)

test_data = utility.create_tensors(test_data, word2idx, tokenizer, max_length)

test_data_with_labels = utility.append_truth(
    test_data, ground_truth_test, test_ids)

embed = utility.load_embedding(embedding_path, word2idx, dim)
embed = torch.tensor(embed)

network = cnn.CNN1D(embed, freeze=False)
network.load_state_dict(torch.load(model_path))
network.eval()
network.to(device)

for item, id in zip(test_data_with_labels, test_ids):
    input = item[0]
    label = item[1]

    input = input.unsqueeze(0)

    out = network(input)
    out = torch.softmax(out,dim=1)
    print(f"User {id} | Confidence in being fake: {out[0][1]} | Label: {ground_truth_test[id]}")