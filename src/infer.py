from typing import List

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.WDNodeMPNN import WDNodeMPNN
from src.featurization import poly_smiles_to_graph
from Analyse_Results import visualize_results

def infer(smiles_list: List[str], ea_list: List[float], ip_list: List[float], model, device = torch.device("cpu")):
    assert type(model) == str or type(model) == WDNodeMPNN

    graphs = []
    valid_indexes = []

    N = range(len(smiles_list))
    for i, smiles, ea, ip  in tqdm(zip(N, smiles_list, ea_list, ip_list), total=len(N), desc="creating graphs"):
        try:
            graph = poly_smiles_to_graph(smiles, [ea], [ip])
            graphs.append(graph)
            valid_indexes.append(i)

        except Exception as e:
            print(e)
            print(f"{smiles} is not valid polymer string")

    if type(model) == str:
        model_temp = WDNodeMPNN(133, 14, hidden_dim=300)
        model_temp.load_state_dict(torch.load(model))
        model = model_temp

    loader = DataLoader(dataset=graphs,
               batch_size=16, shuffle=False)

    pred = []

    for batch in tqdm(loader, total=len(loader), desc="processing batches"):
        out = model(batch)
        pred += out.tolist()

    visualize_results(pred, ea_list, label='ea', save_folder='../Data/figs')
    visualize_results(pred, ip_list, label='ip', save_folder='../Data/figs')


if __name__ == "__main__":
    df = pd.read_csv("../Data/dataset-poly_chemprop.csv")
    # print(df.columns)
    df = df.sample(1000)

    smiles = df['poly_chemprop_input'].tolist()
    ea = df['EA vs SHE (eV)'].tolist()
    ip = df['IP vs SHE (eV)'].tolist()

    infer(smiles, ea, ip, '../Data/model_ea.pt')
