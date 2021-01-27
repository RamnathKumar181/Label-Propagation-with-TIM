import torch
import random
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
from torch_geometric.data import Data

from copy import deepcopy
import numpy as np
from scipy import sparse
from torch_scatter import scatter

import shutil
import os

import numpy as np
np.random.seed(0)

''' Set Random Seed '''
def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_folder(name, model):
    model_dir = f'../configs/{name}'

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    with open(f'{model_dir}/metadata', 'w') as f:
        f.write(f'# of params: {sum(p.numel() for p in model.parameters())}\n')
    return model_dir


def sgc(x, adj, num_propagations):
    for _ in tqdm(range(num_propagations)):
        x = adj @ x
    return torch.from_numpy(x).to(torch.float)


''' Label Propagation '''
def lp(adj, train_idx, labels, num_propagations, p, alpha, preprocess):
    if p is None:
        p = 0.6
    if alpha is None:
        alpha = 0.4

    c = labels.max() + 1
    idx = train_idx
    y = np.zeros((labels.shape[0], c))
    y[idx] = F.one_hot(labels[idx],c).numpy().squeeze(1)
    result = deepcopy(y)
    for i in tqdm(range(num_propagations)):
        result = y + alpha * adj @ (result**p)
        result = np.clip(result,0,1)
    return torch.from_numpy(result).to(torch.float)

''' Diffusion '''
def diffusion(x, adj, num_propagations, p, alpha):
    if p is None:
        p = 1.
    if alpha is None:
        alpha = 0.5

    inital_features = deepcopy(x)
    x = x **p
    for i in tqdm(range(num_propagations)):
#         x = (1-args.alpha)* inital_features + args.alpha * adj @ x
        x = x - alpha * (sparse.eye(adj.shape[0]) - adj) @ x
        x = x **p
    return torch.from_numpy(x).to(torch.float)


''' Community detection feature '''
def community(data, post_fix):
    print('Setting up community detection feature')
    np_edge_index = np.array(data.edge_index)

    G = nx.Graph()
    G.add_edges_from(np_edge_index.T)

    partition = community_louvain.best_partition(G)
    np_partition = np.zeros(data.num_nodes)
    for k, v in partition.items():
        np_partition[k] = v

    np_partition = np_partition.astype(np.int)

    n_values = int(np.max(np_partition) + 1)
    one_hot = np.eye(n_values)[np_partition]

    result = torch.from_numpy(one_hot).float()

    torch.save( result, f'../configs/embeddings/community{post_fix}.pt')

    return result

''' Spectral diffusion '''
def spectral(data, post_fix):
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./norm_spec.jl")
    print('Setting up spectral embedding')
    data.edge_index = to_undirected(data.edge_index)
    np_edge_index = np.array(data.edge_index.T)


    N = data.num_nodes
    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csr')
    result = torch.tensor(Main.main(adj, 128)).float()
    torch.save(result, f'../configs/embeddings/spectral{post_fix}.pt')

    return result


''' Pre-process '''
def preprocess(data, preprocess = "diffusion", num_propagations = 10, p = None, alpha = None, use_cache = True, post_fix = ""):
    if use_cache:
        try:
            x = torch.load(f'../configs/embeddings/{preprocess}{post_fix}.pt')
            print('Using cache')
            return x
        except:
            print(f'../configs/embeddings/{preprocess}{post_fix}.pt not found or not enough iterations! Regenerating it now')

    if preprocess == "community":
        return community(data, post_fix)

    if preprocess == "spectral":
        return spectral(data, post_fix)


    print('Computing adj...')
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    adj = adj.to_scipy(layout='csr')

    sgc_dict = {}

    print(f'Start {preprocess} processing')

    if preprocess == "sgc":
        result = sgc(data.x.numpy(), adj, num_propagations)
#     if preprocess == "lp":
#         result = lp(adj, data.y.data, num_propagations, p = p, alpha = alpha, preprocess = preprocess)
    if preprocess == "diffusion":
        result = diffusion(data.x.numpy(), adj, num_propagations, p = p, alpha = alpha)

    torch.save(result, f'../configs/embeddings/{preprocess}{post_fix}.pt')

    return result


''' Set device '''
def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, dict):
        return type(x)({key: to_device(val, device) for key, val in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([to_device(item, device) for item in x])
    elif isinstance(x, torch.nn.Module):
        return x.to(device)
    else:
        raise NotImplementedError
