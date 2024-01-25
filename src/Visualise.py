from sklearn.manifold import TSNE
import math
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

# %% visualise


def visualize(dict_loader, model, device, label):
    batches = list(range(len(dict_loader)))

    # now lets plot prediction vs true
    store_pred = []
    store_true = []
    # loop over test_loader batches, calculate NN prediction and get true label
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
    plt.scatter(store_true, store_pred, s=1, c=z)
    plt.plot(np.arange(min(store_true)-0.5, max(store_true)+1.5, 1),
             np.arange(min(store_true)-0.5, max(store_true)+1.5, 1), 'r', linewidth=1)
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.grid()
    plt.title('GNN prediction vs true label')
    #plt.set(adjustable='box', aspect='equal')
    plt.text(-4.4, 1, f'R2 = {R2:>3f}', fontsize=10)
    plt.text(-4.7, 0.7, f'RMSE = {RMSE:>3f}', fontsize=10)

    return fig, R2, RMSE

# %% Compare Weights


def compare_weights(weights_1, weights_2):
    '''
    A function to compare weights between two models
    '''
    keys1 = weights_1.keys()
    keys2 = weights_2.keys()
    common_keys = [key for key in keys1 if key in keys2]

    for keys in common_keys:
        print(
            f'{keys} : percent of element change = {torch.sum(torch.eq(weights_1[keys], weights_2[keys])/torch.numel(weights_1[keys]))}, root mean squared difference of elements  = {torch.sqrt(torch.sum((weights_1[keys] - weights_2[keys])**2)/(torch.numel(weights_1[keys])))}')
# %%
