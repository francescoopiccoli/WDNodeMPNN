# Node-centred WDMPNN

## Description
This is a node-centred version of the wd-mpnn model proposed by Aldeghi and Coley in the paper 'A graph representation of molecular ensembles for polymer property prediction' (2022). 

The model obtains the same performance of the edge-centred original version, while greatly simplifying the code, and the need to precompute the huge and sparsem `edge x edge` and `edge x node` adjacency matrices, hence allowing minibatch shuffling and a cleaner, more elegant and computationally efficient solution. 

The featurization (data preprocessing) and the visualization code was taken from the work done from Tammo Dukker on 'Self-Supervised Graph Neural Networks for Polymer Property Prediction'.

## Results
![Results](https://raw.githubusercontent.com/francescoopiccoli/WDNodeMPNN/improvements/results.png?token=GHSAT0AAAAAACKBUN5QNFUXAXPI33LTRA5SZNXTJUA)

## Setup
   ```bash
   git clone https://github.com/francescoopiccoli/WDNodeMPNN.git
   cd WDNodeMPNN
   conda create --name wdncmpnn --file env.yml
   conda activate wdncmpnn
```


To train the model run:
```
python main.py
```


To change the predicted property, set this line to `'EA'` in `main.py`:
```
property = 'IP'
```

To run hyperparamater optimization, set `should_optimize` variable to `True` in `main.py`.
