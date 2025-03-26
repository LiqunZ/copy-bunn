from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.utils import get_laplacian, to_dense_adj
import torch
from torch_geometric.data import Data
from numpy.linalg import eigh


def gen_er_data(n, p, dim, num_graphs=1, spectral=False):
    data_ls = []
    for _ in range(num_graphs):
        edge_index = erdos_renyi_graph(n, p)
        x = torch.rand([n, dim])
        if spectral:
            edge_ind, edge_weight = get_laplacian(edge_index, normalization='sym')
            L = to_dense_adj(edge_ind, edge_attr=edge_weight, max_num_nodes=n).numpy()
            eig_val, eig_vec = eigh(L)
            eig_val, eig_vec = torch.from_numpy(eig_val[0]).unsqueeze(0), torch.from_numpy(eig_vec[0])  # [n, 1], [n, n]
            data = Data(x=x, edge_index=edge_index, eig_vals=eig_val, eig_vecs=eig_vec)
        else:
            data = Data(x=x, edge_index=edge_index)
        data_ls += [data]
    return data_ls