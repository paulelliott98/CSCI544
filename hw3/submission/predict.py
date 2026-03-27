import torch
import argparse
import os
from model import NERModel
from utils import load_dev_data, load_test_data, get_case_feature
from config import UNK_IDX

def parse_args():    
    def valid_file(path):
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError(f"File does not exist: {path}")
        return path
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=valid_file)
    parser.add_argument("--model", required=True, type=valid_file)
    parser.add_argument("--output", required=True)

    return parser.parse_args()

def encode_sentences(sentences, word_to_idx, use_case=False):
    """
    Encodes sentences into word IDs and optional case IDs
    """
    encoded = []

    for words, _ in sentences:
        word_ids = []
        case_ids = []

        for word in words:
            idx = word_to_idx.get(word, word_to_idx.get(word.lower(), UNK_IDX))
            word_ids.append(idx)

            if use_case:
                case_ids.append(get_case_feature(word))

        if use_case:
            encoded.append((word_ids, case_ids))
        else:
            encoded.append(word_ids)

    return encoded


def predict(model, encoded_sentences, idx_to_tag, use_case):
    """
    Predict tags for sentences using model
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for sent in encoded_sentences:
            if use_case:
                word_ids, case_ids = sent
                inputs_word = torch.tensor(word_ids).unsqueeze(0)
                inputs_case = torch.tensor(case_ids).unsqueeze(0)
                outputs = model(inputs_word, inputs_case)
            else:
                inputs_word = torch.tensor(sent).unsqueeze(0)
                outputs = model(inputs_word)

            preds = torch.argmax(outputs, dim=-1)
            pred_tags = [idx_to_tag[i] for i in preds[0].tolist()]
            predictions.append(pred_tags)

    return predictions

def write_predictions(sentences, predictions, output_file):
    with open(output_file, "w") as f:
        zipped = list(zip(sentences, predictions))
        for ind, ((words, _), preds) in enumerate(zipped):
            for i, (word, label) in enumerate(zip(words, preds), start=1):
                f.write(f"{i} {word} {label}\n")

            if ind < len(zipped) - 1:
                f.write("\n")

def main():
    args = parse_args()
    model_path, data_path, output_path = args.model, args.data, args.output

    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    word_to_idx = checkpoint["word_to_idx"]
    tag_to_idx = checkpoint["tag_to_idx"]
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    use_case = checkpoint.get("use_case", False)

    # Load embedding matrix from checkpoint (if it exists)
    embedding_matrix = checkpoint.get("embedding_matrix", None)

    # Initialize model
    model = NERModel(
        vocab_size=len(word_to_idx),
        num_tags=len(tag_to_idx),
        embedding_weights=embedding_matrix,
        use_case=use_case
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load and encode data
    try:
        sentences = load_dev_data(data_path)
    except ValueError:
        sentences = load_test_data(data_path)
        
    encoded = encode_sentences(sentences, word_to_idx, use_case=use_case)

    # Predict
    predictions = predict(model, encoded, idx_to_tag, use_case)

    # Write output
    write_predictions(sentences, predictions, output_path)
    print(f"Predictions written to {output_path}")

if __name__ == "__main__":
    main()