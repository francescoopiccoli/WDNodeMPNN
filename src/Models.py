# # %% Packages
# import torch
# from torch.nn import Sequential, ReLU, Linear
# from torch_geometric.nn import MessagePassing, global_mean_pool

# # %% Define Message Passing Layers


# class Lin_layer_MP(torch.nn.Module):
#     '''
#     Linear NN used in the weighted edge centered MP from scratch
#     '''

#     def __init__(self, in_channels=300, out_channels=300):
#         super(Lin_layer_MP, self).__init__()

#         # Define linear layer and relu
#         self.lin = Linear(in_channels, out_channels)
#         self.relu = ReLU()

#     def forward(self, h0, weighted_sum):
#         x = self.lin(weighted_sum)
#         h1 = self.relu(h0 + x)

#         return h1


# class vec_scratch_MP_layer(torch.nn.Module):
#     '''
#     This is a vectorized implementation of the edge centered message passing
#     '''

#     def __init__(self, in_channels=300, out_channels=300):
#         super(vec_scratch_MP_layer, self).__init__()

#         # Define linear layer and relu
#         self.lin = Linear(in_channels, out_channels)
#         self.relu = ReLU()

#     def forward(self, h0, dest_is_origin_matrix, dev):
#         # pass weighted sum through a NN to obtain new featurization of that edge
#         weighted_sum = torch.sparse.mm(
#             dest_is_origin_matrix.to(dev), h0.to(dev))
#         x = self.lin(weighted_sum)
#         h_next = self.relu(h0+x)

#         return h_next


# class Lin_layer_node(torch.nn.Module):
#     '''
#     Linear NN used in the weighted atom updater
#     '''

#     def __init__(self, in_channels, out_channels=300):
#         super(Lin_layer_node, self).__init__()

#         # Define linear layer and relu
#         self.lin = Sequential(Linear(in_channels, out_channels),
#                               ReLU())

#     def forward(self, concat_atom_edges):
#         atom_hidden = self.lin(concat_atom_edges)

#         return atom_hidden


# class vec_atom_updater(torch.nn.Module):
#     '''
#     This is a vectorized version of the atom update step
#     '''

#     def __init__(self, in_channels, out_channels=300):
#         super(vec_atom_updater, self).__init__()

#         # Define linear layer and relu
#         self.lin = Sequential(Linear(in_channels, out_channels),
#                               ReLU())

#     def forward(self, nodes, h, inc_edges_to_atom_matrix, device):
#         sum_inc_edges = torch.sparse.mm(inc_edges_to_atom_matrix.to(device), h)
#         atom_embeddings = torch.cat((nodes.to(device), sum_inc_edges), dim=1)
#         # pass through NN
#         atom_updates = self.lin(atom_embeddings)
#         return atom_updates

# # %% Define models


# class wMPNN(torch.nn.Module):
#     '''
#     This is the main Weighted Message Passing model. In the forward pass we first get the hidden edge features, then do edge centered message passing,
#     then update atom features, pool the atom features to get molecular fingerprint and then finally pass through three linear layers for regression
#     '''

#     def __init__(self, node_dim, edge_dim, device, hidden_dim=300, classifier_dim=50, output_dim=1):
#         super(wMPNN, self).__init__()

#         # define parameters
#         self.node_dim = node_dim
#         self.edge_dim = edge_dim
#         self.hidden_dim = hidden_dim

#         # define a linear layer to obtain initial hidden features
#         self.lin1 = Sequential(Linear(node_dim+edge_dim, hidden_dim),
#                                ReLU(),
#                                ).to(device)

#         # define edge message passing layers
#         self.vec_scratch_MP_layer1 = vec_scratch_MP_layer(
#             in_channels=hidden_dim, out_channels=hidden_dim).to(device)
#         self.vec_scratch_MP_layer2 = vec_scratch_MP_layer(
#             in_channels=hidden_dim, out_channels=hidden_dim).to(device)
#         self.vec_scratch_MP_layer3 = vec_scratch_MP_layer(
#             in_channels=hidden_dim, out_channels=hidden_dim).to(device)
#         # define node message passing layer
#         self.vec_atom_updater = vec_atom_updater(
#             in_channels=node_dim+hidden_dim).to(device)

#         # Define regression linear layers
#         self.ANN1 = Linear(hidden_dim, classifier_dim).to(device)
#         self.ANN2 = Linear(classifier_dim, classifier_dim).to(device)
#         self.ANN3 = Linear(classifier_dim, output_dim).to(device)
#         self.relu = ReLU().to(device)

#     def forward(self, graph, dest_is_origin_matrix, inc_edges_to_atom_matrix, device):
#         # get parameters
#         nodes = graph.x
#         edge_index = graph.edge_index
#         edge_attr = graph.edge_attr
#         atom_weights = graph.W_atoms

#         # Repeat the node features for each edge
#         nodes_to_edge = nodes[edge_index[0]]
#         # Initialize the edge feautres with the concatenation of the node and edge features
#         h0 = torch.cat([nodes_to_edge, edge_attr], dim=1)
#         # Pass this through a NN to compute the initialize hidden feautres
#         h0 = self.lin1(h0)

#         # pass the messages along edges
#         h1 = self.vec_scratch_MP_layer1(h0, dest_is_origin_matrix, device)
#         h2 = self.vec_scratch_MP_layer2(h1, dest_is_origin_matrix, device)
#         h3 = self.vec_scratch_MP_layer3(h2, dest_is_origin_matrix, device)

#         # get atom embeddings by summing over all incoming edges and concatenating with origanl atom features
#         atom_embeddings = self.vec_atom_updater(
#             nodes, h3, inc_edges_to_atom_matrix, device)

#         # readout layer -> get molecular fingerprint
#         x = global_mean_pool(
#             atom_embeddings * atom_weights.view(-1, 1), graph.batch)

#         # Apply a final regression.
#         out_lin1 = self.ANN1(x)
#         out_lin1 = self.relu(out_lin1)
#         out_lin2 = self.ANN2(out_lin1)
#         out_lin2 = self.relu(out_lin2)
#         out = self.ANN3(out_lin2)

#         return out, x
