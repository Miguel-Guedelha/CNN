from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch
import torch.optim as optim
import utility
import cnn
from timeit import default_timer as timer

dim = 300

#Check bottom of file for model saving path

#Choose intended dataset to train on and test with overall accuracy
base = 'data/fake-vs-real-unfiltered/'
#base = 'data/fake-vs-non_fake-unfiltered/'
#base = 'data/pan20/'

train_dir = base + 'train/'
test_dir = base + 'test/'
embedding_path = f'embeddings/crawl-300d-2M.vec'

#Type of testing to be made:
#   n-grams = 0
#   user mentions = 1
#   hashtags = 2
type = 2

#Number of training epochs, can be changed or the program can simply be stopped when a good enough val/test accuracy is reached 
epochs = 20
#baseline accuracy to save a model dict to file
max = 50

seed_value = 42

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

train_data, train_ids = utility.load_and_process(train_dir, type)
ground_truth_train = utility.load_truth(train_dir + 'truth.txt')

test_data, test_ids = utility.load_and_process(test_dir, type)
ground_truth_test = utility.load_truth(test_dir + 'truth.txt')

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

vocab_data = train_data + test_data

word2idx, max_length = utility.build_vocab(vocab_data, tokenizer)

train_data = utility.create_tensors(
    train_data, word2idx, tokenizer, max_length)

test_data = utility.create_tensors(test_data, word2idx, tokenizer, max_length)

train_data_with_labels = utility.append_truth(
    train_data, ground_truth_train, train_ids)

test_data_with_labels = utility.append_truth(
    test_data, ground_truth_test, test_ids)

df_train = utility.create_frame(train_data_with_labels, train_ids)

df_test = utility.create_frame(test_data_with_labels, test_ids)


print("Loading embedding")
start = timer()
#embed = utility.load_binary_embedding(embedding_path)
embed = utility.load_embedding(embedding_path, word2idx, dim)
embed = torch.tensor(embed)
end = timer()
print(f"Embeddings loaded. Time taken: {round(end-start, ndigits=1)} seconds")


network = cnn.CNN1D(embed, freeze=False)
network.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(network.parameters(), lr=0.25, rho=0.9)

for epoch in range(epochs):
    start = timer()
    df_train = df_train.sample(frac=1, random_state=epoch)
    i = 1
    batch = []
    batch_truths = []
    for index, row in df_train.iterrows():
        inputs = row['data']
        label = row['truth']
        batch.append(inputs)
        batch_truths.append(label)
        optimizer.zero_grad()
        # Batches of 20
        if i % 20 == 0:
            batch = torch.stack(batch).to(device)
            batch_truths = torch.tensor(batch_truths).long().to(device)
            output = network(batch)
            loss = criterion(output, batch_truths)
            loss.backward()
            optimizer.step()
            i = 1
            batch = []
            batch_truths = []
        else:
            i += 1

    end = timer()
    print(
        f'Finished training for {epoch+1} epochs after {round(end-start,ndigits=1)} seconds')

    ###########################################

    predictions, actuals = list(), list()

    network.eval()
    batch = []
    batch_ids = []
    i = 1
    for index, row in df_test.iterrows():
        inputs = row['data']
        label = row['truth']
        batch.append(inputs)
        batch_truths.append(label)

        if i % 20 == 0:

            batch = torch.stack(batch).to(device)
            batch_truths = torch.tensor(batch_truths).long().to(device)

            output = network(batch)
            output = torch.argmax(output, dim=1)
            for i in range(output.shape[0]):
                predictions.append(output[i])
                actuals.append(batch_truths[i])

            i = 1
            batch = []
            batch_truths = []
        else:
            i += 1

    results = zip(predictions, actuals)

    right = 0
    total = len(predictions)
    for predict, actual in results:
        if predict == actual:
            right += 1

    acc = round((right/total)*100, 2)

    print(f"Overall test accuracy for {epoch+1} epochs: {acc}%")

    if acc > max:
        
        '''
        #If testing ngrams on PAN data
        torch.save(network.state_dict(
        ), f'models/pan_v2.pth')
        '''

        '''
        #If testing ngrams on covid fake vs real
        torch.save(network.state_dict(
        ), f'models/covid_fake_real_v2.pth')
        '''

        
        #If testing hashtags on covid fake vs real
        torch.save(network.state_dict(
        ), f'models/covid_fake_real_v2_hashtags.pth')
        

        '''
        #If testing mentions on covid fake vs real
        torch.save(network.state_dict(
        ), f'models/covid_fake_real_v2_mentions.pth')
        '''


        '''
        #If testing n-grams on covid fake vs non_fake
        torch.save(network.state_dict(
        ), f'models/covid_fake_non_fake_v2.pth')
        '''

        '''
        #If testing hashtags on covid fake vs non_fake
        torch.save(network.state_dict(
        ), f'models/covid_fake_non_fake_v2_hashtags.pth')
        '''

        '''
        #If testing mentions on covid fake vs non_fake
        torch.save(network.state_dict(
        ), f'models/covid_fake_non_fake_v2_mentions.pth')
        '''

        max = acc
    network.train()
