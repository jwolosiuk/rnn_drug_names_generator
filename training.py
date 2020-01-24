import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from livelossplot import PlotLosses


from preprocessing import get_names, get_derivatives

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

names = dataset = get_names()
char2id, id2char = get_derivatives(names)

MAX_NAME_LENGTH = 50


def encode(name, end=True):
    BEGIN_ID = len(char2id)
    AFTER_ID = len(char2id) + 1

    code = [char2id[c] for c in name.lower()]
    if end:
        return torch.tensor([BEGIN_ID] + code + [AFTER_ID]).unsqueeze(0)
    else:
        return torch.tensor([BEGIN_ID] + code).unsqueeze(0)

def prepare_dataset():
    max_len = MAX_NAME_LENGTH
    BEGIN_ID = len(char2id)
    AFTER_ID = len(char2id) + 1

    X = np.zeros((len(names), max_len), dtype=np.int64)
    X[:, :] = AFTER_ID
    X[:, 0] = BEGIN_ID

    Y = np.zeros(len(names), dtype=np.int64)

    for i, name in enumerate(names):
        Y[i] = 0
        for j, c in enumerate(name):
            if j + 1 >= max_len:
                break
            X[i, j + 1] = char2id[c]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    trainloader = DataLoader(TensorDataset(torch.from_numpy(X_train).long(),
                                           torch.from_numpy(Y_train).long()),
                             batch_size=128, shuffle=True)
    testloader = DataLoader(TensorDataset(torch.from_numpy(X_test).long(),
                                          torch.from_numpy(Y_test).long()),
                            batch_size=128, shuffle=False)

    dataloaders = {
        "train": trainloader,
        "validation": testloader
    }
    return dataloaders


def train_model_gener(model, criterion, optimizer, dataloaders, num_epochs=10):
    liveloss = PlotLosses()
    model = model.to(device)

    for epoch in range(num_epochs):
        logs = {}
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs_full, labels_class in dataloaders[phase]:

                # here are changes!
                inputs = inputs_full[:, :-1].to(device)
                labels = inputs_full[:, 1:].to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            prefix = ''
            if phase == 'validation':
                prefix = 'val_'

            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()

        liveloss.update(logs)
        liveloss.draw()

class GenerativeLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        dictionary_len = len(char2id) + 2
        self.emb = nn.Embedding(dictionary_len, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size)
        # note: input size is the numer of channels/embedding dim, NOT length
        self.fc = nn.Linear(hidden_size, dictionary_len)

    def forward(self, x):
        x = self.emb(x)
        x = x.permute(1, 0, 2)  # BLC -> LBC
        output, (hidden, cell) = self.lstm(x)
        res = self.fc(output)
        return res.permute(1, 2, 0) #  LBC -> BCL

def get_trained_model(num_epochs=100):
    model = GenerativeLSTM(embedding_size=5, hidden_size=16)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    dataloaders = prepare_dataset()
    train_model_gener(model, criterion, optimizer, dataloaders, num_epochs=num_epochs)
    return model
