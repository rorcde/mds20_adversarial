import torch
import torch.nn as nn


class Deep_lev(torch.nn.Module):

    def __init__(self, vocab_size=30522, embedding_dim=128, hidden_dim=128) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 3, 1)


    def encode_sequence(self, sequence):
        embedded_sequence = self.embeddings(sequence)
        out, (ht, ct) = self.encoder(embedded_sequence)
        return ht[-1]

    def get_embeddings(self, onehots_a, onehots_b):
        embs_a = torch.stack([torch.matmul(v, self.embeddings.weight) for v in onehots_a])
        
        return embs_a

    def forward_on_embeddings(self, sequence_a, sequence_b):
        embs_a = self.get_embeddings(sequence_a, sequence_b)
        _, (embedded_sequence_a, _) = self.encoder(embs_a)
        embedded_sequence_a= embedded_sequence_a[-1]
        
        embedded_sequence_b = self.encode_sequence(sequence_b)
        diff = torch.abs(embedded_sequence_a - embedded_sequence_b)
        representation = torch.cat([embedded_sequence_a, embedded_sequence_b, diff], dim=-1)
        approx_distance = self.linear(representation)

        return approx_distance


    def forward(self, sequence_a, sequence_b):
        embedded_sequence_a = self.encode_sequence(sequence_a)
        embedded_sequence_b = self.encode_sequence(sequence_b)
        diff = torch.abs(embedded_sequence_a - embedded_sequence_b)
        representation = torch.cat([embedded_sequence_a, embedded_sequence_b, diff], dim=-1)

        approx_distance = self.linear(representation)

        return approx_distance
