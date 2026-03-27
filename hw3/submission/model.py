import torch
from torch import nn
from config import *

class NERModel(nn.Module):
    def __init__(self, vocab_size, num_tags, embedding_weights=None, use_case=False):
        super().__init__()
        self.use_case = use_case

        if embedding_weights is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                embedding_weights,
                freeze=False,
                padding_idx=PAD_IDX
            )
        else: # default initialization
            self.word_embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=EMBEDDING_DIM,
                padding_idx=PAD_IDX
            )

        if use_case:
            # initialize case embeddings
            self.case_embedding = nn.Embedding(4, CASE_EMBEDDING_DIM)
            lstm_input_size = EMBEDDING_DIM + CASE_EMBEDDING_DIM
        else:
            lstm_input_size = EMBEDDING_DIM

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT
        )

        self.linear = nn.Linear(
            LSTM_HIDDEN_DIM * 2,
            LINEAR_OUTPUT_DIM
        )

        self.elu = nn.ELU()

        self.classifier = nn.Linear(
            LINEAR_OUTPUT_DIM,
            num_tags
        )

    def forward(self, word_ids, case_ids=None):
        word_embeddings = self.word_embedding(word_ids)

        if self.use_case:
            case_embeddings = self.case_embedding(case_ids)
            x = torch.cat([word_embeddings, case_embeddings], dim=-1)
        else:
            x = word_embeddings

        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.elu(x)
        x = self.classifier(x)
        return x