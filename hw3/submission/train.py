import argparse
import os
import inspect
from seqeval.metrics import f1_score
from typing import cast
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from model import NERModel
from config import *
from utils import load_glove_gz, load_dev_data, build_embedding_matrix, get_case_feature

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def parse_args():
    def valid_file_glove(path):
        if path is None:
            return path
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError(f"File does not exist: {path}")
        return path
    
    def valid_file(path):
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError(f"File does not exist: {path}")
        return path
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True, type=valid_file)
    parser.add_argument("--dev", required=True, type=valid_file)
    parser.add_argument("--glove", default=None, type=valid_file_glove)
    parser.add_argument("--model", default=None, type=valid_file)
    return parser.parse_args()

def collate_fn(batch):
    word_ids, case_ids, tag_ids = zip(*batch)

    longest = max(len(s) for s in word_ids)

    padded_sentences = []
    padded_cases = []
    padded_tags = []

    for s, c, t in zip(word_ids, case_ids, tag_ids):
        pad_length = longest - len(s)

        padded_sentences.append(s + [PAD_IDX] * pad_length)
        padded_cases.append(c + [0] * pad_length)
        padded_tags.append(t + [PAD_IDX] * pad_length)

    return torch.tensor(padded_sentences), torch.tensor(padded_cases), torch.tensor(padded_tags)

def encode_sentence(sentence, word_to_idx):
    word_ids = []
    case_ids = []

    for word in sentence:
        word_ids.append(word_to_idx.get(word, word_to_idx["<UNK>"]))
        case_ids.append(get_case_feature(word))

    return word_ids, case_ids

def encode_tags(tags, tag_to_idx):
    return [tag_to_idx[t] for t in tags]

def evaluate_f1(model, dev_loader, tag_to_idx, device):
    model.eval()
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}

    sig = inspect.signature(model.forward)
    expects_case = "case_ids" in sig.parameters

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dev_loader:
            if expects_case:
                word_ids, case_ids, labels = batch
                word_ids = word_ids.to(device)
                case_ids = case_ids.to(device)
                outputs = model(word_ids, case_ids)
            else:
                word_ids, _, labels = batch
                word_ids = word_ids.to(device)
                outputs = model(word_ids)

            labels = labels.to(device)
            preds = torch.argmax(outputs, dim=-1)

            preds = preds.cpu().tolist()
            labels = labels.cpu().tolist()

            for pred_seq, label_seq in zip(preds, labels):
                sent_preds = []
                sent_labels = []
                for p, l in zip(pred_seq, label_seq):
                    if l != PAD_IDX:
                        sent_preds.append(idx_to_tag[p])
                        sent_labels.append(idx_to_tag[l])
                if sent_labels:
                    all_preds.append(sent_preds)
                    all_labels.append(sent_labels)

    f1 = cast(float, f1_score(all_labels, all_preds, average="micro"))
    return f1

def main():
    args = parse_args()

    train_filepath, dev_filepath, glove_filepath, model_filepath = args.train, args.dev, args.glove, args.model
    glove = load_glove_gz(glove_filepath) if glove_filepath is not None else None

    """Parse dataset"""
    train_data = load_dev_data(path=train_filepath)
    dev_data = load_dev_data(path=dev_filepath)

    """Build vocabularies for words and tags"""
    if model_filepath is not None:
        checkpoint = torch.load(model_filepath, weights_only=False)
        word_to_idx = checkpoint["word_to_idx"]
        tag_to_idx = checkpoint["tag_to_idx"]
    else: 
        word_to_idx = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        tag_to_idx = {}

        for words, tags in train_data:
            for w in words:
                if w not in word_to_idx:
                    word_to_idx[w] = len(word_to_idx)

            for t in tags:
                if t not in tag_to_idx:
                    tag_to_idx[t] = len(tag_to_idx)

    """Create encoded dataset that uses indices"""
    train_encoded = []

    for words, tags in train_data:
        word_ids, case_ids = encode_sentence(words, word_to_idx)
        tag_ids = encode_tags(tags, tag_to_idx)
        train_encoded.append((word_ids, case_ids, tag_ids))

    dev_encoded = []

    for words, tags in dev_data:
        word_ids, case_ids = encode_sentence(words, word_to_idx)
        tag_ids = encode_tags(tags, tag_to_idx)
        dev_encoded.append((word_ids, case_ids, tag_ids))

    """Padding and batching: Pad sequences to same length, then batch with DataLoader"""
    train_dataset = NERDataset(train_encoded)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(42)
    )

    dev_dataset = NERDataset(dev_encoded)

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    """Train model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if glove is None:
        embedding_matrix = None
        use_case = False
    else: # task 2
        embedding_matrix = build_embedding_matrix(
            word_to_idx,
            glove,
            EMBEDDING_DIM
        )
        use_case = True

    new_model_filepath = model_filepath if model_filepath is not None else ("blstm1.pt" if glove is None else "blstm2.pt")

    model = NERModel(
        vocab_size=len(word_to_idx), 
        num_tags=len(tag_to_idx),
        embedding_weights=embedding_matrix,
        use_case=use_case
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX
    )
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=REDUCE_LR_MODE, patience=REDUCE_LR_PATIENCE, factor=REDUCE_LR_FACTOR
    )

    best_f1 = 0

    # If model file specified, load states
    if model_filepath is not None:
        checkpoint = torch.load(model_filepath, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_f1 = checkpoint.get("best_f1", 0)

    print(f'EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, MOMENTUM={MOMENTUM}, REDUCE_LR_FACTOR={REDUCE_LR_FACTOR}')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for word_ids, case_ids, labels in train_loader:
            word_ids = word_ids.to(device)
            case_ids = case_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward pass
            outputs = model(word_ids, case_ids)

            # flatten for cross entropy loss
            outputs = outputs.reshape(-1, len(tag_to_idx))
            labels = labels.reshape(-1)

            # compute loss
            loss = criterion(outputs, labels)

            # compute gradients
            loss.backward()

            # update model parameters using computed gradients
            optimizer.step()

            total_loss += loss.item()
        
        f1 = evaluate_f1(model, dev_loader, tag_to_idx, device)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "word_to_idx": word_to_idx,
                "tag_to_idx": tag_to_idx,
                "best_f1": best_f1,
                "embedding_matrix": embedding_matrix,
                "use_case": use_case
            }, new_model_filepath)  # save best model
            print(f"Model saved to {new_model_filepath}")
        
        scheduler.step(f1)

if __name__ == "__main__":
    main()