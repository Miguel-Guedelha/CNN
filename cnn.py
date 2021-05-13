import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def create_emb_layer(weight, freeze=False):
    num_embeddings, embedding_dim = weight.shape
    emb_layer = nn.Embedding.from_pretrained(weight, freeze=freeze)

    return emb_layer, embedding_dim


class CNN1D(nn.Module):
    def __init__(self, weights, freeze=False):
        super().__init__()
        # CONV1D.V2
        self.vocab_size, self.embedding_dim = weights.shape
        self.embedding = nn.Embedding.from_pretrained(weights, freeze)
        self.kernel_sizes = [3, 4, 5]

        if self.embedding_dim == 300:
            t = int(self.embedding_dim/3)
        elif self.embedding_dim >= 100:
            t = int(self.embedding_dim/2)
            self.filters = [t] * 3
        else:
            t = self.embedding_dim

        self.filters = [t] * 3

        self.conv1d_list = nn.ModuleList(
            [nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.filters[i], kernel_size=self.kernel_sizes[i]) for i in range(len(self.kernel_sizes))])
        self.fc = nn.Linear(np.sum(self.filters), 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):

        in_embed = self.embedding(input).float()

        in_reshaped = in_embed.permute(0, 2, 1)

        out_conv_list = [F.relu(conv1d(in_reshaped))
                         for conv1d in self.conv1d_list]

        out_pool_list = [F.max_pool1d(out_conv, kernel_size=out_conv.shape[2])
                         for out_conv in out_conv_list]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in out_pool_list],
                         dim=1)

        logits = self.fc(self.dropout(x_fc))

        return logits
