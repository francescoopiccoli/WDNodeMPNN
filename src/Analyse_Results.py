'''
Here I plot the performance and create the figures used in the thesis
'''


# %% Packages
from Models import *
from sklearn.manifold import TSNE
import math
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
from scipy.stats import gaussian_kde

# %% Call train loader
dict_test_loader = torch.load('dict_test_loader.pt')
num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features
device = 'cpu'

# %% Call wmPNN
# create an instance of the model
model = wMPNN(num_node_features, num_edge_features, device)

# %% Call the trained weights

label = 0

data_dir = f'/Users/tammodukker/Desktop/Python Thesis Master/Run_Results/RUNS_sup_EA'
models_weights = [torch.load(os.path.join(
    data_dir, f'Run_{i}/Models_Weights/supervised_weights_data=1.pt'), map_location=torch.device('cpu')) for i in range(1, 11)]
models = [model.load_state_dict(model_state_dict, strict=True)
          for model_state_dict in models_weights]
# %% visualis


def visualize_models(dict_loader, models_weights, device, label):
    batches = list(range(len(dict_loader)))

    # now lets plot prediction vs true
    store_pred = []
    store_true = []

    # loop over test_loader batches, calculate NN prediction and get true label
    for i, weights in enumerate([models_weights[1]]):
        model.load_state_dict(weights, strict=True)
        for batch in batches:
            data = dict_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)

            model.eval()
            model.to(device)

            pred = model(data, dest_is_origin_matrix,
                         inc_edges_to_atom_matrix, device)[0]
            pred_cpu = torch.Tensor.cpu(pred).flatten().detach().numpy()

            if label == 0:
                true = torch.Tensor.cpu(data.y1)
            elif label == 1:
                true = torch.Tensor.cpu(data.y2)

            true_cpu = true.detach().numpy()
            store_pred.append(pred_cpu)
            store_true.append(true_cpu)

    # flatten all batches into one list
    store_pred = [item for sublist in store_pred for item in sublist]
    store_true = [item for sublist in store_true for item in sublist]
    xy = np.vstack([store_pred, store_true])
    z = gaussian_kde(xy)(xy)

    # calculate R2 score and RMSE
    R2 = r2_score(store_true, store_pred)
    RMSE = math.sqrt(mean_squared_error(store_true, store_pred))

    # now lets plot
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    plt.scatter(store_true, store_pred, s=5, c=z)
    plt.plot(np.arange(min(store_true)-0.5, max(store_true)+1.5, 1),
             np.arange(min(store_true)-0.5, max(store_true)+1.5, 1), 'r--', linewidth=1)
    plt.xlabel('True (eV)')
    plt.ylabel('Prediction (eV)')
    plt.grid()
    plt.title(f'Electron Affinity')
    #plt.set(adjustable='box', aspect='equal')
    plt.text(-4.1, 0.5, f'R2 = {R2:.3f}', fontsize=10)
    plt.text(-4.5, 0.2, f'RMSE = {RMSE:.3f}', fontsize=10)

    return fig, R2, RMSE


# %% Compare Weights
visualize_models(dict_test_loader, models_weights, device, label)
# %%
