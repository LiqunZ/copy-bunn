from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data

from torch import from_numpy

from torch.nn.functional import one_hot

from torch_geometric.datasets.graph_generator import GridGraph
from math import cos, sin
import torch


import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.dense import floyd_warshall_numpy
from scipy.spatial import distance_matrix


class DistTransform:
    def __init__(self, deg: int = 1):
        """
        Initialization for the DistTransform class. Computes the dense matrix of shape (V, V) where the i, j entry is
        the (geodesic) distance between two points in the graphs raised to a (possibly negative) power, which is the
        degree parameter.

        Args:
            deg: power to raise the entries of the distance matrix
        """
        self.deg = deg

    def transform(self, data: Data) -> Data:
        """
        Args:
            data: graph representated as a Data object
        """

        return compute_distances(data, deg=self.deg)

    def __str__(self):
        """
        String format of the class used for saving to disk without loss of information.
        """
        return f"DistTransform-power={self.deg}"


def compute_distances(data, deg: float, max_val=2):
    """From a torch_geometric Data object computes the dense distance matrix"""
    G = to_networkx(data)
    A = floyd_warshall_numpy(G) # dense distance matrix
    if deg < 0:
        A_deg = from_numpy(A).pow(deg)
        A_deg = A_deg.nan_to_num(posinf=max_val)  # the nans are the distance 0 nodes
    else:
        A_deg = from_numpy(A).pow(deg)
    max_row = A_deg.sum(axis=0).max()  # if bounded input then bounded output
    data.dist_mat = 1/max_row*A_deg.reshape((-1, 1))
    data.enc = None
    return data


def generate_dist_data(train_data, num_gen=100):
    data_list = []
    for data in train_data:
        enc = torch.rand((data.num_nodes, 10))
        for _ in range(num_gen):
            x = (torch.rand((data.num_nodes, 1)) + -0.5) * 3  # between -1.5 and 1.5
            dist_mat = data.dist_mat.reshape(data.num_nodes, data.num_nodes).float()
            y = dist_mat @ x
            data_cp = Data(x=x, edge_index=data.edge_index, y=y, dist_mat=dist_mat.reshape((-1, 1)), enc=enc)
            data_list += [data_cp]
    return data_list


def generate_curved_line_data(num_samples, length=10, theta=0, unique_id=False):
    # generates a list of Data objects with x, egde_index and ys
    shf = SheafNN()

    data_ls = []
    line_graph = GridGraph(height=length, width=1)()  # 10 nodes with self loops
    line_graph.bundle = gen_bundle(line_graph.edge_index, theta).reshape([-1, 4])
    # unique_id = torch.rand([10, 2]) - 0.5
    if unique_id:
        enc = one_hot(torch.arange(0, 10)).float()
    for _ in range(num_samples):
        x = torch.rand([length, 2]) - 0.5
        y = x
        for _ in range(128):
            y = shf(x=y, edge_index=line_graph.edge_index, rests=line_graph.bundle)
        if unique_id:
            data_ls += [Data(x=x, edge_index=line_graph.edge_index, bundle=line_graph.bundle, y=y, enc=enc)]
        else:
            data_ls += [Data(x=x, edge_index=line_graph.edge_index, bundle=line_graph.bundle, y=y)]

    return data_ls


def gen_bundle(edge_index, theta=0):
    bdl_ls = []
    for i, j in edge_index.transpose(0, 1).tolist():
        if i < j:
            Oij = [[cos(theta), sin(theta)],
                   [-sin(theta), cos(theta)]]
        elif j < i:
            Oij = [[cos(-theta), sin(-theta)],
                   [-sin(-theta), cos(-theta)]]
        else:
            Oij = [[1, 0],
                   [0, 1]]
        bdl_ls += [Oij]
    return torch.tensor(bdl_ls, dtype=torch.float)




def get_many_tile(num_samples, tile):
    data_ls = []
    for num in np.arange(11, tile+1, 10):
        for _ in range(num_samples):
            data = GridGraph(num, 1)()  # chain graph, 1d grid
            pos = np.expand_dims(np.linspace(0, 1, num=num), 1)

            data.pos = torch.tensor(pos, dtype=torch.float)

            dist = distance_matrix(pos, pos)  # can be done more efficient

            ker = (dist < 0.1 + 1e-6).astype(float)  # threshold if the distance is smaller than 0.1
            num_neighs = ker.sum(axis=0)  # in order to normalize
            ker_norm = np.diag(np.power(num_neighs, -1)) @ ker  # operator computing the mean over the neighborhoud at distance < 0.1

            signal = np.random.rand((num))  # random input features
            y = torch.tensor(ker_norm @ signal, dtype=torch.float).unsqueeze(1)  # mean of signal over the neighborhoud
            enc = torch.tensor(pos, dtype=torch.float)
            x = torch.tensor(signal, dtype=torch.float).unsqueeze(1)
            x = torch.cat((x, enc), dim=1)  # cat to keep all info
            data.x = x
            data.y = y
            data_ls += [data]
    return data_ls

def get_tile(num_samples, tile):
    data_ls = []
    num = tile
    for _ in range(num_samples):
        data = GridGraph(num, 1)()  # linegraph
        pos = np.expand_dims(np.linspace(0, 1, num=num), 1)

        data.pos = torch.tensor(pos, dtype=torch.float)

        dist = distance_matrix(pos, pos)  # can be done more efficient

        ker = (dist < 0.1 + 1e-6).astype(float)
        num_neighs = ker.sum(axis=0)
        ker_norm = np.diag(np.power(num_neighs, -1)) @ ker  # operator computing the mean over the neighborhoud at distance < 0.1

        signal = np.random.rand((num))  # random input features
        y = torch.tensor(ker_norm @ signal, dtype=torch.float).unsqueeze(1)
        enc = torch.tensor(pos, dtype=torch.float)
        x = torch.tensor(signal, dtype=torch.float).unsqueeze(1)
        x = torch.cat((x, enc), dim=1)  # cat to keep all info
        data.x = x
        data.y = y
        data_ls += [data]
    return data_ls


