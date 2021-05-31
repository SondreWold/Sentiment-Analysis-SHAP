import os
import argparse
import time
import json
import numpy as np
import torch
from transformers import AdamW, BertForSequenceClassification
import tqdm
from torch.utils.data import DataLoader
from torch import nn
from typing import Dict, Any
from dataset import NorecDataset
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter


def multi_acc(y_pred, y_test):  # Taken from: https://github.com/dh1105/Sentence-Entailment
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1)
           == y_test).sum().float() / float(y_test.size(0))
    return acc


DEFAULT_CONFIG = {
    "name": "default",
}


def parse_args() -> argparse.Namespace:
    '''
    Command line interface for launching experiments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path of json file for configuration")
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()
    return args


def write_tensorboard(writer, log_foler, epoch, train_loss, val_loss, train_acc, val_acc) -> None:
    try:
        os.makedirs(log_foler)
    except OSError as e:
        if(epoch == 0):
            print("Log folder already exist!")

    writer.add_scalar("training loss", train_loss, epoch)
    writer.add_scalar("validation loss", val_loss, epoch)
    writer.add_scalar("training accuracy", train_acc, epoch)
    writer.add_scalar("validation accuracy", val_acc, epoch)


def save_model(folder: str, name: str, epoch: int, model, train_time_sum, parameters) -> None:

    try:
        os.makedirs(folder)
    except OSError as e:
        if(epoch == 0):
            print("Folder already exist!")

    path = folder + name + ".pt"

    torch.save({
        "name": name,
        "epoch": epoch,
        "train_time_sum": train_time_sum,
        "parameters": parameters
    }, path)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(folder)

    print(f"Saved model to {folder}")


def train_bert_model(train_dataset, val_dataset, config: Dict[str, Any], test_run) -> nn.Module:
    """
    Trains model and checkpoints on best validation loss
    """
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    folder = config["folder"]
    name = config["name"]
    log_folder = config["log_folder"]
    fine_tuning = config["fine_tuning"]
    bert_variant = config.get("bert_variant", "bert-base-uncased")
    checkpointing_enabled = config["checkpointing_enabled"]
    writer = SummaryWriter(log_folder)

    print(f"Selected BERT variant was: {bert_variant}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    # Declare components and datasets
    tokenizer = BertTokenizer.from_pretrained(bert_variant)
    model = BertForSequenceClassification.from_pretrained(
        bert_variant, num_labels=2)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)

    if fine_tuning == "decay":
        print("Using the decay fine-tuning strategy")
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    elif fine_tuning == "frozen":
        print("All params frozen")
        for param in model.base_model.parameters():
            param.requires_grad = False
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        print("Standard setting. Allowing all parameters to fine-tune.")
        optimizer = AdamW(model.parameters(), lr=lr)

    best_val_loss = 999999
    best_val_acc = 0
    train_time_sum = 0
    model = model.to(dev)
    for epoch in range(0, epochs):
        print("\n")
        print(f"----- Starting epoch: {epoch} -----")
        start_epoch_time = time.time()
        total_train_loss = 0
        total_train_acc = 0
        total_val_loss = 0
        total_val_acc = 0
        training_passes = 0
        validation_passes = 0

        # Training loop
        model.train()
        for i, batch in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            doc, y = batch
            doc = list(doc)
            X = tokenizer(doc, padding=True,
                          return_tensors="pt")
            input_ids = X["input_ids"].to(dev)
            token_type_ids = X["token_type_ids"].to(dev)
            attention_mask = X["attention_mask"].to(dev)
            y = y.to(dev)

            loss, prediction = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                     attention_mask=attention_mask, labels=y.squeeze()).values()

            total_train_loss += loss.item()
            acc = multi_acc(prediction, y.squeeze())
            total_train_acc += acc.item()
            loss.backward()
            optimizer.step()
            training_passes += 1
            if(test_run):
                break

        # Validation loop
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(val_loader)):
                optimizer.zero_grad()
                doc, y = batch
                doc = list(doc)
                X = tokenizer(doc, padding=True,
                              return_tensors="pt")
                input_ids = X["input_ids"].to(dev)
                token_type_ids = X["token_type_ids"].to(dev)
                attention_mask = X["attention_mask"].to(dev)
                y = y.to(dev)
                loss, prediction = model(
                    input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=y.squeeze()).values()

                total_val_loss += loss.item()
                acc = multi_acc(prediction, y.squeeze())
                total_val_acc += acc.item()
                validation_passes += 1
                if(test_run):
                    break

        # Book-keeping
        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time
        train_loss = total_train_loss/training_passes
        train_acc = total_train_acc/training_passes
        val_loss = total_val_loss/validation_passes
        val_acc = total_val_acc/validation_passes
        train_time_sum += epoch_time

        # Tensorboard
        write_tensorboard(writer, log_folder, epoch,
                          train_loss, val_loss, train_acc, val_acc)

        print(f"Epoch: {epoch}. Training loss: {train_loss}, Val loss: {val_loss}, Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}, epoch time: {epoch_time}")

        # Checkpointing
        if val_loss < best_val_loss and checkpointing_enabled:
            parameters = sum(
                p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"Val loss is lower than previous best which was at {best_val_loss}. Saving model, with number of parameters: {parameters}")
            save_model(folder, name, epoch, model, train_time_sum, parameters)
            best_val_loss = val_loss

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if(test_run):
            break

    print(
        f"Finished training in {train_time_sum} seconds. Best val loss was: {best_val_loss}. Best val acc was: {best_val_acc}.")


def set_random_seeds(seed: int) -> None:
    '''
    Sets random seed.
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config: Dict[str, Any]) -> None:
    '''
    Main program loop.
    args:
        config(dict): a configuration file for the experiment.
    '''

    print("Starting training phase of model: ")
    print(config["name"])

    train_dataset = NorecDataset(config["train_path"])
    val_dataset = NorecDataset(config["dev_path"])

    train_bert_model(train_dataset, val_dataset, config, args.test_run)


if __name__ == "__main__":
    args = parse_args()

    config = DEFAULT_CONFIG
    config.update(vars(args))

    if args.config:
        with open(args.config, "r") as f:
            config.update(json.load(f))

    seed = config["seed"]

    print("Running with random seed:", seed)
    set_random_seeds(seed)
    main(config)
