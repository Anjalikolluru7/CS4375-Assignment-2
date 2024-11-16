import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        hidden_layer = self.activation(self.W1(input_vector))
        predicted_vector = self.softmax(self.W2(hidden_layer))
        return predicted_vector


def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {word: idx for idx, word in enumerate(vocab_list)}
    index2word = {idx: word for idx, word in enumerate(vocab_list)}
    vocab.add(unk)
    return vocab, word2index, index2word


def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]

    return tra, val


def train_epoch(model, optimizer, data, minibatch_size):
    model.train()
    correct = 0
    total = 0
    epoch_loss = 0
    random.shuffle(data)
    for minibatch_index in tqdm(range(len(data) // minibatch_size)):
        minibatch = data[minibatch_index * minibatch_size: (minibatch_index + 1) * minibatch_size]
        input_vectors = torch.stack([item[0] for item in minibatch])
        gold_labels = torch.tensor([item[1] for item in minibatch])
        optimizer.zero_grad()
        predicted_vectors = model(input_vectors)
        loss = model.compute_Loss(predicted_vectors, gold_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        correct += (predicted_vectors.argmax(1) == gold_labels).sum().item()
        total += len(gold_labels)
    return epoch_loss / (len(data) // minibatch_size), correct / total


def evaluate(model, data, minibatch_size):
    model.eval()
    correct = 0
    total = 0
    epoch_loss = 0
    with torch.no_grad():
        for minibatch_index in tqdm(range(len(data) // minibatch_size)):
            minibatch = data[minibatch_index * minibatch_size: (minibatch_index + 1) * minibatch_size]
            input_vectors = torch.stack([item[0] for item in minibatch])
            gold_labels = torch.tensor([item[1] for item in minibatch])
            predicted_vectors = model(input_vectors)
            loss = model.compute_Loss(predicted_vectors, gold_labels)
            epoch_loss += loss.item()
            correct += (predicted_vectors.argmax(1) == gold_labels).sum().item()
            total += len(gold_labels)
    return epoch_loss / (len(data) // minibatch_size), correct / total


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    test_data = None
    if args.test_data != "to fill":
        print("========== Loading test data ==========")
        test_data, _ = load_data(args.test_data, args.test_data)
        test_data = convert_to_vector_representation(test_data, word2index)

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, optimizer, train_data, minibatch_size=16)
        val_loss, val_acc = evaluate(model, valid_data, minibatch_size=16)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Time taken: {time.time() - start_time:.2f}s")

    if test_data:
        test_loss, test_acc = evaluate(model, test_data, minibatch_size=16)
        print("========== Test Results ==========")
        print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")