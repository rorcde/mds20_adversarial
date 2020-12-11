import torch
import Levenshtein as Lev
from typing import Sequence, Dict, Any, List

def calculate_wer(sequence_a: str, sequence_b: str) -> int:
    # taken from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
    b = set(sequence_a.split() + sequence_b.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in sequence_a.split()]
    w2 = [chr(word2char[w]) for w in sequence_b.split()]

    return Lev.distance(''.join(w1), ''.join(w2))

class SequenceModifier:
    # taken from https://github.com/fursovia/dilma/blob/ef582e73a1cfce5f6bd753fb08b588f12f5cde8f/adat/utils.py#L103
    def __init__(
            self,
            vocab: List[str],
            remove_prob: float = 0.05,
            add_prob: float = 0.05,
            replace_prob: float = 0.1
    ) -> None:
        assert sum([remove_prob, add_prob, replace_prob]) > 0.0
        self.vocab = vocab
        self.remove_prob = remove_prob
        self.add_prob = add_prob
        self.replace_prob = replace_prob

    def remove_token(self, sequence: List[str]) -> List[str]:
        samples = np.random.binomial(n=1, p=self.remove_prob, size=len(sequence))
        sequence = [token for i, token in enumerate(sequence) if not samples[i]]
        return sequence

    def replace_token(self, sequence: List[str]) -> List[str]:
        samples = np.random.binomial(n=1, p=self.replace_prob, size=len(sequence))
        new_sequence = [random.choice(self.vocab) if samples[i] else sequence[i] for i in range(len(sequence))]
        return new_sequence

    def add_token(self, sequence: List[str]) -> List[str]:
        new_sequence = sequence + [
            random.choice(self.vocab)
            for _ in range(np.random.binomial(len(sequence), self.add_prob))
        ]
        return new_sequence

    def __call__(self, sequence: str) -> str:
        splitted_sequence = sequence.split()
        if len(splitted_sequence) > 1 and self.remove_prob:
            splitted_sequence = self.remove_token(splitted_sequence)

        if self.replace_prob:
            splitted_sequence = self.replace_token(splitted_sequence)

        if self.add_prob:
            splitted_sequence = self.add_token(splitted_sequence)
        return " ".join(splitted_sequence)
