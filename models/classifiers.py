import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    '''
    Convolutional Neural Network for text classification based on [1].
    Credits for implementation: [2]
    Inputs:
        vocab_size - size of vocabulary of dataset used
        emb_dim - embedding space dimension
        n_filters - number of filters
        filter_sizes - sizes of convolution filters
        output_dim - dimension of the output (number of classes in our case)
        dropout - drop out probability
        pad_idx - padding index for sequences
    [1]: Yoon Kim. Convolutional neural networks for sentence classification. (2014)
    [2]: https://github.com/etomoscow/DL-in-NLP/blob/master/hw3/task4_sentiment_cnn.ipynb
    '''

    def __init__(self, vocab_size, emb_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def get_embeddings(self, onehots):
        embs = torch.stack([torch.matmul(v, self.embedding.weight) for v in onehots])
        return embs

    def forward_on_embeddings(self, inputs):
        embs = self.get_embeddings(inputs)
        out = [F.relu(c(embs.permute(0, 2, 1))) for c in self.convs]
        out_pool = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in out]
        cat = self.dropout(torch.cat(out_pool, dim=1))
        final = self.fc(cat)
        return final

    def forward(self, text):
        embs = self.embedding(text)
        embs = embs.permute(0, 2, 1)
        out = [F.relu(c(embs)) for c in self.convs]
        out_pool = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in out]
        cat = self.dropout(torch.cat(out_pool, dim=1))
        final = self.fc(cat)
        return final


class TextGRU(nn.Module):
    '''
    Recurrent Neural Network based on Gated Rectified Unit for text classification [1].
    Credits for implementation: [2]
    Inputs:
        vocab_size - size of vocabulary of dataset used
        emb_dim - embedding space dimension
        hidden_dim - dimension of hidden state of RNN
        output_dim - dimension of the output (number of classes in our case)
        dropout - drop out probability
        pad_idx - padding index for sequences
    [1]: Empirical evaluation of gated recurrent neural networks on sequence modeling, 2014.
    [2]: https://github.com/etomoscow/DL-in-NLP/blob/master/hw3/task3_sentiment_rnn.ipynb
    '''

    def __init__(self, vocab_size, emb_dim, hidden_dim, out_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=1, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def get_embeddings(self, onehots):
        embs = torch.stack([torch.matmul(v, self.embedding.weight) for v in onehots])
        return embs

    def forward_on_embeddings(self, inputs):
        embs = self.get_embeddings(inputs)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embs, text_lengths, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)
