import numpy as np


def binary_encoder(description, embedding_dim=512):
    padding_char = '\0'
    description = description + (embedding_dim - len(description)) * padding_char
    description = bytes(description, 'utf-8')
    description = np.frombuffer(description, dtype=np.int8)
    description = np.asarray(description, np.float32)
    return description


def universe_sentence_encoder(description, embedding_dim=512):
    raise NotImplementedError
