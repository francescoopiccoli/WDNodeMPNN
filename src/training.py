import torch
from featurization import poly_smiles_to_graph
import pandas as pd
import random
from torch_geometric.loader import DataLoader
from WDNodeMPNN import WDNodeMPNN
import os
import tqdm

# %% Hyperparameters
device = 'cpu'
hyper_params = {
    'batch_size': 64,
    'epochs': 100,
    'hidden_dimension': 300,
    'learning_rate': 1e-3,
}

# %% Call data
df = pd.read_csv('../Data/dataset-poly_chemprop.csv')

graphs = []
# check if graphs_list.pt exists
if not os.path.isfile('Graphs_list.pt'):
    print('Creating Graphs_list.pt')
    for i in range(len(df.loc[:, 'poly_chemprop_input'])):
        poly_input = df.loc[i, 'poly_chemprop_input']
        poly_label1 = df.loc[i, 'EA vs SHE (eV)']
        poly_label2 = df.loc[i, 'IP vs SHE (eV)']
        graph = poly_smiles_to_graph(poly_input, poly_label1, poly_label2) # given the input polymer string, this function returns a pyg data object
        graphs.append(graph)
        if i % 100 == 0:
            print(f"[{i} / {len(df.loc[:, 'poly_chemprop_input'])}]")

    torch.save(graphs, 'Graphs_list.pt')
else:
    graphs = torch.load('Graphs_list.pt')

# shuffle graphs
random.seed(12345)
data_list_shuffle = random.sample(graphs, len(graphs))

# take 80-20 split for training - test data
train_datalist = data_list_shuffle[:int(0.8*len(data_list_shuffle))]
test_datalist = data_list_shuffle[int(0.8*len(data_list_shuffle)):]

# print some statistics
print(f'Number of training graphs: {len(train_datalist)}')
print(f'Number of test graphs: {len(test_datalist)}')

num_node_features = train_datalist[0].num_node_features
num_edge_features = train_datalist[0].num_edge_features
print(f'Number of node features: {num_node_features}')
print(f'Number of edge features:{num_edge_features} ')


batch_size = hyper_params['batch_size']

train_loader = DataLoader(dataset=train_datalist,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_datalist,
                         batch_size=batch_size, shuffle=False)


epochs = hyper_params['epochs']
hidden_dimension = hyper_params['hidden_dimension']

model = WDNodeMPNN(num_node_features, num_edge_features, hidden_dim=hidden_dimension)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])
criterion = torch.nn.MSELoss()

def train(loader, label):
    model.train()
    total_loss = 0.0
    for data in loader:
        out = model(data)
        # Calculate the loss based on the specified label.
        if label == 0:
            loss = criterion(out, data.y1.float())
        elif label == 1:
            loss = criterion(out, data.y2.float())

        loss.backward()  
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return model, total_loss / len(loader)

    
def test(loader, label):
    model.eval()
    total_loss = 0
    for data in loader:
        out = model(data)
        if label == 0:
            loss = criterion(out, data.y1.float())
        elif label == 1:
            loss = criterion(out, data.y2.float())

        total_loss += loss.item()
    return total_loss / len(loader)


# %% Train model
for epoch in tqdm.tqdm(range(epochs)):
    model, train_loss = train(train_loader, 0)
    test_loss = test(test_loader, 0)
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# %% Save model
torch.save(model.state_dict(), 'model.pt')




# # check that it works, each batch has one big graph
# for step, data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}', '\n')
#     print(data)
#     print()
#     if step == 1:
#         break

# %% Create Matrices needed for Message Passing


# def MP_Matrix_Creator(loader):
#     '''
#     Here we create the two matrices needed later for the message passinng part of the graph neural network. 
#     They are in essence different forms of the adjacency matrix of the graph. They are created on a batch thus the batches cannot be shuffled
#     The graph and both matrices are saved per batch in a dictionary
#     '''
#     dict_graphs_w_matrix = {}
#     for batch, graph in enumerate(loader):
#         # get attributes of graphs in batch
#         nodes = graph.x
#         edge_index = graph.edge_index
#         edge_attr = graph.edge_attr
#         atom_weights = graph.W_atoms
#         bond_weights = graph.W_bonds
#         num_bonds = edge_index[0].shape[0]

#         '''
#         Create edge update message passing matrix
#         '''
#         dest_is_origin_matrix = torch.zeros(
#             size=(num_bonds, num_bonds)).to(device)
#         # for sparse matrix
#         I = torch.empty(2, 0, dtype=torch.long)
#         V = torch.empty(0)

#         for i in range(num_bonds):
#             # find edges that are going to the originating atom (neigbouring edges)
#             incoming_edges_idx = (
#                 edge_index[1] == edge_index[0, i]).nonzero().flatten()
#             # check whether those edges originate from our bonds destination atom, if so ignore that bond
#             idx_from_dest_atom = (
#                 edge_index[0, incoming_edges_idx] == edge_index[1, i])
#             incoming_edges_idx = incoming_edges_idx[idx_from_dest_atom != True]
#             # find the features and assoociated weights of those neigbouring edges
#             weights_inc_edges = bond_weights[incoming_edges_idx]
#             # create matrix
#             dest_is_origin_matrix[i, incoming_edges_idx] = weights_inc_edges

#             # For Sparse Version
#             edge = torch.tensor([i])
#             # create indices
#             i1 = edge.repeat_interleave(len(incoming_edges_idx))
#             i2 = incoming_edges_idx.clone()
#             i = torch.stack((i1, i2), dim=0)
#             # find assocociated values
#             v = weights_inc_edges

#             # append to larger arrays
#             I = torch.cat((I, i), dim=1)
#             V = torch.cat((V, v))

#         # create a COO sparse version of edge message passing matrix
#         dest_is_origin_sparse = torch.sparse_coo_tensor(
#             I, V, [num_bonds, num_bonds])
#         '''
#         Create node update message passing matrix
#         '''
#         inc_edges_to_atom_matrix = torch.zeros(
#             size=(nodes.shape[0], edge_index.shape[1])).to(device)

#         I = torch.empty(2, 0, dtype=torch.long)
#         V = torch.empty(0)
#         for i in range(nodes.shape[0]):
#             # find index of edges that are incoming to specific atom
#             inc_edges_idx = (edge_index[1] == i).nonzero().flatten()
#             weights_inc_edges = bond_weights[inc_edges_idx]
#             inc_edges_to_atom_matrix[i, inc_edges_idx] = weights_inc_edges

#             # for sparse version
#             node = torch.tensor([i])
#             i1 = node.repeat_interleave(len(inc_edges_idx))
#             i2 = inc_edges_idx.clone()
#             i = torch.stack((i1, i2), dim=0)
#             v = weights_inc_edges

#             I = torch.cat((I, i), dim=1)
#             V = torch.cat((V, v))

#         # create a COO sparse version of node message passing matrix
#         inc_edges_to_atom_sparse = torch.sparse_coo_tensor(
#             I, V, [nodes.shape[0], edge_index.shape[1]])

#         if batch % 10 == 0:
#             print(f"[{batch} / {len(loader)}]")

#         '''
#         Store in Dictionary
#         '''
#         dict_graphs_w_matrix[str(batch)] = [
#             graph, dest_is_origin_sparse, inc_edges_to_atom_sparse]

#     return dict_graphs_w_matrix


# # %% Create dictionary with bathed graphs and message passing matrices for supervised train set
# dict_train_loader = MP_Matrix_Creator(train_loader)
# torch.save(dict_train_loader, 'dict_train_loader.pt')
# # %% Create dictionary with bathed graphs and message passing matrices for test set
# dict_test_loader = MP_Matrix_Creator(test_loader)
# torch.save(dict_test_loader, 'dict_test_loader.pt')


# print('Done')


# %% Test making Sparse Matrices

# get one batch
# for batch, graphs in enumerate(train_loader):
#     graph = graphs
#     break

# # get attributes of graphs in batch
# nodes = graph.x
# edge_index = graph.edge_index
# edge_attr = graph.edge_attr
# atom_weights = graph.W_atoms
# bond_weights = graph.W_bonds
# num_bonds = edge_index[0].shape[0]
# dest_is_origin_matrix = torch.zeros(
#     size=(num_bonds, num_bonds)).to(device)

'''
Create edge update message passing matrix
'''
# I = torch.empty(2, 0, dtype=torch.long)
# V = torch.empty(0)

# for i in range(num_bonds):
#     # find edges that are going to the originating atom (neigbouring edges)
#     incoming_edges_idx = (edge_index[1] == i).nonzero().flatten()
#     # check whether those edges originate from our bonds destination atom, if so ignore that bond
#     idx_from_dest_atom = (
#         edge_index[0, incoming_edges_idx] == edge_index[1, i])
#     incoming_edges_idx = incoming_edges_idx[idx_from_dest_atom != True]
#     # find the features and assoociated weights of those neigbouring edges
#     weights_inc_edges = bond_weights[incoming_edges_idx]
#     # create matrix
#     dest_is_origin_matrix[i, incoming_edges_idx] = weights_inc_edges

#     # For Sparse Version
#     edge = torch.tensor([i])
#     i1 = edge.repeat_interleave(len(incoming_edges_idx))
#     i2 = incoming_edges_idx.clone()
#     i = torch.stack((i1, i2), dim=0)
#     v = weights_inc_edges

#     I = torch.cat((I, i), dim=1)
#     V = torch.cat((V, v))

# # create a COO sparse version of node message passing matrix
# dest_is_origin_sparse = torch.sparse_coo_tensor(I, V, [num_bonds, num_bonds])
# '''
# Create node update message passing matrix
# '''
# inc_edges_to_atom_matrix = torch.zeros(
#     size=(nodes.shape[0], edge_index.shape[1])).to(device)

# I = torch.empty(2, 0, dtype=torch.long)
# V = torch.empty(0)
# for i in range(nodes.shape[0]):
#     # find index of edges that are incoming to specific atom
#     inc_edges_idx = (edge_index[1] == edge_index[0, i]).nonzero().flatten()
#     weights_inc_edges = bond_weights[inc_edges_idx]
#     inc_edges_to_atom_matrix[i, inc_edges_idx] = weights_inc_edges

#     # for sparse version
#     node = torch.tensor([i])
#     i1 = node.repeat_interleave(len(inc_edges_idx))
#     i2 = inc_edges_idx.clone()
#     i = torch.stack((i1, i2), dim=0)
#     v = weights_inc_edges

#     I = torch.cat((I, i), dim=1)
#     V = torch.cat((V, v))

# inc_edges_to_atom_sparse = torch.sparse_coo_tensor(
#     I, V, [nodes.shape[0], edge_index.shape[1]])

# # %% SEE IF THEY ARE EQUAL
# dest_is_origin_dense = dest_is_origin_sparse.to_dense()
# print(torch.equal(dest_is_origin_matrix, dest_is_origin_dense))
# inc_edges_to_atom_dense = inc_edges_to_atom_sparse.to_dense()
# print(torch.equal(inc_edges_to_atom_matrix, inc_edges_to_atom_dense))

# %%

