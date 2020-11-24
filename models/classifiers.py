import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embs = self.embedding(text)
        embs = embs.permute(0, 2, 1)
        out = [F.relu(c(embs)) for c in self.convs]
        out_pool = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in out]
        cat = self.dropout(torch.cat(out_pool, dim=1))
        final = self.fc(cat)
        return final

class GRU_Text(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 hidden_dim,
                 out_dim,
                 dropout,
                 pad_idx
                 ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=pad_idx
            )
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True
            )
        self.fc = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        #text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        #embedded = [sent len, batch size, emb dim]
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.detach().cpu())
        packed_output, hidden = self.rnn(packed_embedded)
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        #hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)
