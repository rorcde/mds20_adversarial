import math
import torch
import torch.nn as nn

class Deep_lev(torch.nn.Module):
    '''
    This `Model` implements a framework for fast approximate similarity search as described in [Convolutional Embedding for Edit Distance](https://arxiv.org/abs/2001.11692).
    The `DeepLevenshtein` takes two sequences (texts) `x` and `y` as inputs outputs an approximated edit-distance score between `x` and `y`.

    # Parameters
    sequance_a.shape = [seq_len, batch_size]
    sequance_b.shape = [seq_len, batch_size]
    approx_distance.shape = [batch_size, 1]
    distance - initial distance between two tokenized sequences

    '''

    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)

        self.linear = nn.Linear(hidden_dim * 3, 1)
        self._loss = nn.MSELoss()


    def encode_sequence(self, sequence):
        embedded_sequence = self.embeddings(sequence)
        out, (ht, ct) = self.encoder(embedded_sequence)
        return ht[-1]

    def forward(self, sequence_a, sequence_b, distance):
        embedded_sequence_a = self.encode_sequence(sequence_a)
        embedded_sequence_b = self.encode_sequence(sequence_b)
        diff = torch.abs(embedded_sequence_a - embedded_sequence_b)
        representation = torch.cat([embedded_sequence_a, embedded_sequence_b, diff], dim=-1)

        approx_distance = self.linear(representation)
        output_dict = {"distance": approx_distance}

        if distance is not None:
            output_dict["loss"] = self._loss(approx_distance.view(-1), distance.view(-1))
        return output_dict

