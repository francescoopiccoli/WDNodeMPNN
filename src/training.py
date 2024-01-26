import math
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error

from featurization import poly_smiles_to_graph
import pandas as pd
import random
from torch_geometric.loader import DataLoader
from WDNodeMPNN import WDNodeMPNN
import os
import pandas as pd
import torch
import tqdm
from src.featurization import poly_smiles_to_graph


def get_graphs():
    graphs = []
    # check if graphs_list.pt exists
    if not os.path.isfile('Graphs_list.pt'):
        print('Creating Graphs_list.pt')
        df = pd.read_csv('../Data/dataset-poly_chemprop.csv')
        # use tqdm to show progress bar
        for i in tqdm.tqdm(range(len(df.loc[:, 'poly_chemprop_input']))):
            poly_strings = df.loc[i, 'poly_chemprop_input']
            poly_labels_EA = df.loc[i, 'EA vs SHE (eV)']
            poly_labels_IP = df.loc[i, 'IP vs SHE (eV)']
            # given the input polymer string, this function returns a pyg data object
            graph = poly_smiles_to_graph(
                poly_strings=poly_strings, 
                poly_labels_EA=poly_labels_EA, 
                poly_labels_IP=poly_labels_IP
            ) 

            
            graphs.append(graph)
            
        torch.save(graphs, 'Graphs_list.pt')
        print('Graphs_list.pt saved')
    else:
        print('Loading Graphs_list.pt')
        graphs = torch.load('Graphs_list.pt')

    return graphs


def train(model, loader, label, optimizer, criterion):

    model.train()
    total_loss = 0.0
    for data in loader:
        out = model(data)
        # Calculate the loss based on the specified label.
        if label == 0: # EA
            loss = criterion(out, data.y_EA.float())
        elif label == 1: # IP
            loss = criterion(out, data.y_IP.float())

        loss.backward()  
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return model, total_loss / len(loader)

    
def test(model, loader, label, criterion):
    model.eval()
    total_loss = 0.0
    for data in loader:
        out = model(data)
        if label == 0: # EA
            loss = criterion(out, data.y_EA.float())
        elif label == 1: # IP
            loss = criterion(out, data.y_IP.float())

        total_loss += loss.item()
    return total_loss / len(loader)

# %% Train model
for epoch in tqdm.tqdm(range(epochs)):
    visualize_models(test_loader, [model], 0)
    model, train_loss = train(train_loader, 0)
    test_loss = test(test_loader, 0)
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# %% Save model
torch.save(model.state_dict(), 'model.pt')