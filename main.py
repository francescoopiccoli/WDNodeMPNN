import torch
import random
from torch_geometric.loader import DataLoader
from src.WDNodeMPNN import WDNodeMPNN
import os
import tqdm
from src.training import get_graphs, train, test

# %% Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

hyper_params = {
    'batch_size': 64,
    'epochs': 100,
    'hidden_dimension': 300,
    'learning_rate': 1e-3,
}

labels = { 
    'EA': 0,
    'IP': 1,
}

# %% Load data
graphs = get_graphs()

# shuffle graphs
random.seed(12345)
data_list_shuffle = random.sample(graphs, len(graphs))

# take 80-20 split for training - test data
train_datalist = data_list_shuffle[:int(0.8*len(data_list_shuffle))]
test_datalist = data_list_shuffle[int(0.8*len(data_list_shuffle)):]
num_node_features = train_datalist[0].num_node_features
num_edge_features = train_datalist[0].num_edge_features

# print some statistics
print(f'Number of training graphs: {len(train_datalist)}')
print(f'Number of test graphs: {len(test_datalist)}')
print(f'Number of node features: {num_node_features}')
print(f'Number of edge features:{num_edge_features} ')

batch_size = hyper_params['batch_size']
train_loader = DataLoader(dataset=train_datalist, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_datalist, batch_size=batch_size, shuffle=False)

hidden_dimension = hyper_params['hidden_dimension']

model = WDNodeMPNN(num_node_features, num_edge_features, hidden_dim=hidden_dimension)
model_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))

model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])
criterion = torch.nn.MSELoss()
epochs = hyper_params['epochs']
property = 'EA'
model_save_name = f'{model_name}_{property}.pt'
# %% Train model
for epoch in tqdm.tqdm(range(epochs)):
    model, train_loss = train(model, train_loader, label=labels[property], optimizer=optimizer, criterion=criterion)
    test_loss = test(model, test_loader, label=labels[property], criterion=criterion)

    print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # save model every 10 epochs
    if epoch % 10 == 0:
        if not os.path.isdir('Models'):
            os.mkdir('Models')
        torch.save(model.state_dict(), f'Models/{model_save_name}.pt')
        
# save latest model
torch.save(model.state_dict(), f'Models/{model_save_name}.pt')