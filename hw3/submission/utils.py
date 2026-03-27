import gzip
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def load_glove_gz(path):
    """Load GloVe .gz file and return a dict mapping words to embedding vectors"""
    glove = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = list(map(float, parts[1:]))
            glove[word] = vector
    return glove

def get_case_feature(word):
    if word.isupper():
        return 0
    elif word.istitle():
        return 1
    elif word.islower():
        return 2
    else:
        return 3

def build_embedding_matrix(word_to_idx, glove, embedding_dim):
    vocab_size = len(word_to_idx)

    # random initialization
    embedding_matrix = torch.randn(vocab_size, embedding_dim)

    for word, idx in word_to_idx.items():
        if word == "<PAD>":
            embedding_matrix[idx] = torch.zeros(embedding_dim)
        else:
            glove_vec = glove.get(word.lower())
            if glove_vec is not None:
                embedding_matrix[idx] = torch.tensor(glove_vec)

    return embedding_matrix

def load_dev_data(path):
    sentences = []
    words, tags = [], []

    with open(path) as f:
        for line in f:
            line = line.strip()

            if line == "":
                sentences.append((words, tags))
                words, tags = [], []
            else:
                _, word, tag = line.split()
                words.append(word)
                tags.append(tag)

    if words:
        sentences.append((words, tags))

    return sentences

def load_test_data(path):
    sentences = []
    words = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "":
                if words:
                    sentences.append((words, []))
                    words = []
            else:
                parts = line.split()
                words.append(parts[1])  # just index + word, no tag

    if words:
        sentences.append((words, []))

    return sentences