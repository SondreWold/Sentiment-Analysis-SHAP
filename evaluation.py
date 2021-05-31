import argparse
import torch
import pandas as pd
import json
import numpy as np
from dataset import NorecDataset
from transformers import AdamW, BertForSequenceClassification
import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import classification_report
import scipy as sp


def multi_acc(y_pred, y_test):  # Taken from: https://github.com/dh1105/Sentence-Entailment
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1)
           == y_test).sum().float() / float(y_test.size(0))
    return acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True,
                        help="The path to the test set")
    parser.add_argument("--model_pytorch", type=str, default=None,
                        help="Path to the pytorch model in .pt format")
    parser.add_argument("--bert_path", type=str, default=None,
                        help="Path to the pretrained pytorch model folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_gpu", action="store_true")

    return parser.parse_args()


def eval(device, test_loader, num_labels, tokenizer, path_to_model_folder, batch_size, inverted_label_indexer):
    model = BertForSequenceClassification.from_pretrained(path_to_model_folder)
    model = model.to(device)
    model.eval()
    total_test_acc = 0.
    passes = 0
    guesses = []
    golds = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(test_loader)):
            doc, y = batch
            doc = list(doc)

            X = tokenizer(doc, padding=True, return_tensors="pt")
            input_ids = X["input_ids"].to(device)
            token_type_ids = X["token_type_ids"].to(device)
            attention_mask = X["attention_mask"].to(device)
            y = y.to(device)
            loss, prediction = model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=y.squeeze()).values()

            acc = multi_acc(prediction, y.squeeze())
            guess = [inverted_label_indexer[x] for x in (torch.log_softmax(
                prediction, dim=1).argmax(dim=1)).to("cpu").numpy()]

            gold = [inverted_label_indexer[x]
                    for x in y.squeeze().to("cpu").numpy()]

            print(guess)
            print(gold)

            guesses.extend(guess)
            golds.extend(gold)
            total_test_acc += acc.item()
            print("Batch accuracy: ",  acc.item())
            passes += 1

    print(f"Total accuracy on test test: {total_test_acc/passes}")
    print(classification_report(golds, guesses))


def main(args):
    device = torch.device("cuda" if args.use_gpu else "cpu")
    model_data = torch.load(args.model_pytorch, map_location=device)
    tokenizer = BertTokenizer.from_pretrained("NbAiLab/nb-bert-base")
    name = model_data["name"]
    print(f"Evaluating model: {name}.pt")

    test_dataset = NorecDataset(
        args.test_path)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)
    num_labels = 2

    eval(device, test_loader, num_labels,
         tokenizer, args.bert_path, args.batch_size, {0: "Negative", 1: "Positive"})


if __name__ == "__main__":
    args = parse_args()
    main(args)
