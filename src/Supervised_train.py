# # %% Packages
# import time
# import random
# import Visualise as vs
# from Models import *
# # deep learning packages
# import torch
# from torch.nn import Sequential, ReLU, Linear
# from torch_geometric.nn import global_mean_pool
# import os
# import copy
# import matplotlib.pyplot as plt
# # comet ml
# # Import comet_ml at the top of your file
# from comet_ml import Experiment

# # Create an experiment with your api key
# # experiment = Experiment(
# #     api_key="e5TE1cDV9WuZ9VLAgKyBRxXDv",
# #     project_name="MEP-Supervised-Only",
# #     workspace="tammo117",
# # )

# hyper_params = {
#     'batch_size': 64,
#     'epochs': 100,
#     'hidden_dimension': 300,
#     'learning_rate': 1e-3,
#     'type of run': 'Only Supervised Training'
# }

# # experiment.log_parameters(hyper_params)

# # !!! Which label are we considering 0 = 'EA vs SHE', 1 = 'IP vs SHE'
# label = 0


# # %% GPU
# # setting device on GPU if available, else CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print()

# # Additional Info when using cuda
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

# # %%Hyperparameters
# batch_size = hyper_params['batch_size']
# epochs = hyper_params['epochs']
# hidden_dimension = hyper_params['hidden_dimension']

# # %% Call data

# # dict_train_loader = torch.load('dict_train_loader.pt')
# # dict_test_loader = torch.load('dict_test_loader.pt')

# # num_train_graphs = len(list(dict_train_loader.keys())[
# #     :-2])*batch_size + dict_train_loader[list(dict_train_loader.keys())[-1]][0].num_graphs
# # num_node_features = dict_train_loader['0'][0].num_node_features
# # num_edge_features = dict_train_loader['0'][0].num_edge_features

# # assert dict_train_loader['0'][0].num_graphs == batch_size, 'Batch_sizes of data and model do not match'
# # # check that it works, each batch has one big graph

# # batched_graphs = dict_train_loader['0'][0]
# # print(batched_graphs)
# #print(f'Number of training graphs: {num_train_graphs}')
# #print(f'Number of graphs in the current batch: {batched_graphs.num_graphs}')
# #print(f'Number of node features {num_node_features}')
# #print(f'Number of edge features {num_edge_features}')


# # %% Create an instance of the WMPNN model called from models.py

# model = wMPNN(num_node_features, num_edge_features, device)
# model.to(device)
# # print(model)

# untrained_state_dict = copy.deepcopy(model.state_dict())
# print('Randomly initialized weights')

# # %%# %% Train

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = torch.nn.MSELoss()


# def train(dict_train_loader, size, label):
#     # shuffle batches every epoch
#     order_batches = list(range(len(dict_train_loader)))
#     random.shuffle(order_batches)

#     model.train()
#     # Iterate in batches over the training dataset.
#     for i, batch in enumerate(order_batches):
#         # get graphs & matrices for MP from dictionary
#         data = dict_train_loader[str(batch)][0]
#         data.to(device)
#         dest_is_origin_matrix = dict_train_loader[str(batch)][1]
#         dest_is_origin_matrix.to(device)
#         inc_edges_to_atom_matrix = dict_train_loader[str(batch)][2]
#         inc_edges_to_atom_matrix.to(device)

#         # Perform a single forward pass.
#         out = model(data, dest_is_origin_matrix,
#                     inc_edges_to_atom_matrix, device)[0].flatten()

#         # Compute the loss with specified label.
#         if label == 0:
#             loss = criterion(out, data.y1.float())
#         elif label == 1:
#             loss = criterion(out, data.y2.float())

#         loss.backward()  # Derive gradients.
#         optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.

#         # experiment.log_metric(
#         #     "training_loss"+str(size), loss.item(), step=i)

#         # Print cost every 100 batches and first loss
#         if i % 100 == 0 and i > 0:
#             loss, current = loss.item(), i * batch_size
#             print(
#                 f"loss: {loss:>7f}  [{current:>5d}/{int(list(dict_train_loader.keys())[-1])*batch_size:>5d}]")
#         elif i == 0:
#             print(f'first loss: {loss.item()}')
#     return model


# def test(dict_loader, train_or_test, size, label):
#     batches = list(range(len(dict_loader)))
#     model.eval()
#     test_loss = 0
#     # Iterate in batches over the training/test dataset.
#     for batch in batches:
#         data = dict_loader[str(batch)][0]
#         data.to(device)
#         dest_is_origin_matrix = dict_loader[str(batch)][1]
#         dest_is_origin_matrix.to(device)
#         inc_edges_to_atom_matrix = dict_loader[str(batch)][2]
#         inc_edges_to_atom_matrix.to(device)

#         out = model(data, dest_is_origin_matrix,
#                     inc_edges_to_atom_matrix, device)[0].flatten()

#         # Check against ground-truth labels.
#         if label == 0:
#             MSE = criterion(out, data.y1)
#         elif label == 1:
#             MSE = criterion(out, data.y2)

#         test_loss += MSE.item()

#     # log epoch average error
#     # experiment.log_metric("test_loss"+str(size),
#     #                       test_loss/(len(dict_loader)), step=epoch)
#     # print epoch average error
#     print(
#         f"Epoch average {train_or_test} error: {test_loss / len(dict_loader)}")

#     return MSE

#  # %% Split training data
# #amount_training_data = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
# amount_training_data = [1]

# # %% Supervised run over all datasets sizes over all epochs

# for percentage in amount_training_data:
#     last_batch = round(int(list(dict_train_loader.keys())[-1])*percentage)
#     batches = [str(batch) for batch in range(last_batch)]
#     dict_train_loader_part = {k: dict_train_loader[k] for k in batches}
#     print(
#         f'STARTING SUPERVISED RUN OF {percentage} SUPERVISED DATA')

#     # reset model on every different dataset run
#     model.load_state_dict(untrained_state_dict)
#     model = model.to(device)
#     #  Run over all epoch
#     for epoch in range(0, epochs):
#         print(f"Epoch {epoch+1}\n-------------------------------")
#         t1 = time.time()
#         model = train(dict_train_loader_part, percentage, label)
#         t2 = time.time()
#         print(f'epoch time: {t2-t1}\n')
#         test_error = test(dict_test_loader, 'test', percentage, label)

#     # save model weights after training
#     torch.save(model.state_dict(),
#                'Models_Weights/supervised_weights_data='+str(percentage)+'.pt')
#     # visualise R2 and RMSE of model
#     fig, R2, RMSE = vs.visualize(dict_test_loader, model, device, label)
#     fig.savefig('Figures/Supervised_data='+str(percentage) +
#                 '.png')
#     # experiment.log_image('Figures/Supervised_data='+str(percentage) +
#     #                      '.png')
#     # experiment.log_metrics(
#     #     {'R2_'+str(percentage): R2, 'RMSE_'+str(percentage): RMSE})

#     print('Done!\n')
# # experiment.end()


# # %%
