import logging

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm
from torch_geometric.utils import to_networkx, from_networkx, get_laplacian, to_dense_adj
from numpy.linalg import eigh


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data

def lap_eig_transform(data, pad=500):
    """
    Computes the eigenvectors of normalized Laplacian for the graph defined by data

    Args:
        data (torch_geometric.data.Data): The input data graph object

    Returns:
        torch.Tensor: Effective resistances embedding with shape (num_nodes, 500)
    """
    num_nodes = data.x.shape[0]
    edge_ind, edge_weight = get_laplacian(data.edge_index, normalization='sym', num_nodes=num_nodes)
    L = to_dense_adj(edge_ind, edge_attr=edge_weight, max_num_nodes=num_nodes).numpy()
    eigval, eigvec = eigh(L)
    eigval, eigvec = torch.from_numpy(eigval), torch.from_numpy(eigvec[0])  # [n], [n, n]
    
    zer = torch.zeros([eigvec.shape[0], pad - eigvec.shape[1]])  # need to pad with zeros
    eigvec = torch.cat([eigvec, zer], dim=1)  # [n, pad]
    zer = torch.zeros([1, pad-eigval.shape[1]])
    eigval = torch.cat([eigval, zer], dim=1)  # [pad]
    
    data.eigval, data.eigvec = eigval, eigvec
    return data
