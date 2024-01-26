from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.WDNodeMPNN import WDNodeMPNN
from src.featurization import poly_smiles_to_graph
from src.Analyse_Results import visualize_results

def infer_by_dataloader(loader, model, device):
    out, ea, ip = [], [], []

    model.eval()
    model.to(device)

    for batch in tqdm(loader, total=len(loader), desc="processing batches"):
        out += model(batch.to(device)).cpu().tolist()
        ea += batch.y_EA.tolist()
        ip += batch.y_IP.tolist()

    return out, ea, ip

def infer(smiles_list: List[str], ea_list: List[float], ip_list: List[float], model, device = None, visualize=False) -> np.ndarray:
    assert type(model) == str or type(model) == WDNodeMPNN

    if device is None:
        device = torch.device("cpu")

    graphs = []
    valid_indexes = []

    N = range(len(smiles_list))
    for i, smiles, ea, ip  in tqdm(zip(N, smiles_list, ea_list, ip_list), total=len(N), desc="creating graphs"):
        try:
            graph = poly_smiles_to_graph(smiles, [ea], [ip])
            graphs.append(graph)
            valid_indexes.append(i)

        except Exception as e:
            print(f"{smiles} is not valid polymer string")

    if type(model) == str:
        model_temp = WDNodeMPNN(133, 14, hidden_dim=300)
        model_temp.load_state_dict(torch.load(model))
        model = model_temp
        model.to(device)

    loader = DataLoader(dataset=graphs,
               batch_size=16, shuffle=False)

    out, ea, ip = infer_by_dataloader(loader, model, device)

    print(f"{len(smiles_list) - len(valid_indexes)} failed entries (!)")


    if visualize:
        visualize_results(out, ea, label='ea', save_folder='Results/figs')
        visualize_results(out, ip, label='ip', save_folder='Results/figs')

    pred = np.full(len(smiles_list), None, dtype=object)
    pred[valid_indexes] = out

    return pred    