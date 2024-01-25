In this folder we have the python files to perform polymer property prediction in a pure supervised fashion

Important files:
- dataset-poly_chemprop.csv: this is the polymer dataset in SMILES string format
- function_featurization_own: this is a function that takes the polymer SMILES string and turns them into pytorch geometric data objects
- Transform_Batch_Data.py: This is the main featurization file. It calls function_featurization_own, peforms batching, and creates those adjusted adjacency matrices to perform weighted message passing
- Model.py: this contains the weighted message passing neural network
- supervised_train.py: the main training file
- Run.sh: a run file with which one can obtain results from scratch 

To obtain results (what Run.sh does):
- run Transform_Batch_Data.py, this will create two data files dict_train_loader.pt & dict_test_loader.pt, which are the batched polymer graphs including the adjusted adjaceny matrices stores as pytorch geometric objects
- run supervised_train: this class dict_train_loader.pt & dict_test_loader.pt & model.py and performs training
