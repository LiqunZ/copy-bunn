import torch
from torch import Tensor
import warnings

from torch_geometric.graphgym import cfg

from ..bundle.orthogonal import Orthogonal
# from bnn.utils.lookup import get_norm

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.pool.glob import global_mean_pool
from torch_geometric.utils import to_dense_batch

import torch_geometric.graphgym.register as register

import torch.nn as nn
import torch_sparse
import torch.nn.functional as F
from torch.nn import Module


from torch_geometric.graphgym.register import register_layer

class SumGNNLayer(MessagePassing):
    def __init__(self, dim_in,
                 dim_out):
        super().__init__()

        self.lin_self = nn.Linear(dim_in, dim_out)
        self.lin_neighb = nn.Linear(dim_in, dim_out)
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        h = self.lin_self(x)
        msg = self.lin_neighb(self.propagate(x=x, edge_index=edge_index))
        return h + msg

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SumGNN(Module):

    def __init__(self, dim_in,
                 hidden_dim,
                 dim_out, num_layers):
        super().__init__()

        self.num_layers = num_layers
        self.act = nn.GELU()

        self.enc = nn.Linear(dim_in, hidden_dim)
        self.dec = nn.Linear(hidden_dim, dim_out)

        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(SumGNNLayer(hidden_dim, hidden_dim))  # small dimensionality to reduce nb of params
        self.layers = torch.nn.ModuleList(self.layers)
    def forward(self, x, edge_index):
        h = self.act(self.enc(x))
        for k in range(self.num_layers):
            h = self.act(self.layers[k](h, edge_index)) + h
        return self.dec(h)

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j



class BuNNConv(MessagePassing):
    def __init__(self, in_channels: int,
                 bundle_dim: int,
                 num_bundles: int,
                 num_gnn: int,
                 gnn_dim: int,
                 normalize: bool = True,
                 add_self_loops=True,
                 dropout=0.0,
                 max_deg=4,
                 tau=1,
                 orth_meth="householder",
                 tau_method="fixed"):
        super().__init__()

        assert in_channels % bundle_dim == 0

        self.in_channels = in_channels
        self.bundle_dim = bundle_dim
        self.num_bundles = num_bundles
        self.num_bun_params = int((self.bundle_dim-1) * self.bundle_dim / 2 * self.num_bundles)
        self.gnn_dim = gnn_dim

        self.tau_method = tau_method

        self.dropout = dropout

        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.lin = torch.nn.Linear(in_channels, in_channels)

        self.max_deg = max_deg
        self.tau = tau
        self.orth_meth = orth_meth

        self.num_gnn_layers = num_gnn # cfg.bundle.num_gnn

        # Neural network used to compute the bundle: either a GNN, or an MLP () if num_gnn_layers is set to 0
        self.struct_enc = Module()
        if self.num_gnn_layers > 0:
            self.struct_enc = SumGNN(dim_in=in_channels, hidden_dim=self.gnn_dim, dim_out=self.num_bun_params, num_layers=self.num_gnn_layers)
        else:
            self.struct_enc = torch.nn.Sequential(torch.nn.Linear(in_channels, self.gnn_dim),
                                                    torch.nn.GELU(),
                                                    torch.nn.Linear(self.gnn_dim, self.num_bun_params),
                                                  )

        self.orthogonal = Orthogonal(d=bundle_dim, orthogonal_map=self.orth_meth)

    def forward(self,  x, edge_index):
        """
        x has shape [num_nodes, in_channels]
        edge_index has shape [2, num_edges]
        node_rep has shape [num_nodes, out_channels//bundle_dim, bundle_dim, bundle_dim]
                 and consists of in_channels//bundle_dim orthogonal matrices per node
        """

        edge_ind = edge_index


        deg = torch_geometric.utils.degree(edge_index[0], dtype=edge_index.dtype)
        deg_inv = deg.float().pow(-1)
        col, row = edge_index[0], edge_index[1]
        edge_weight = deg_inv[row]

        num_nodes = x.shape[0]
        vector_field = x.reshape(num_nodes, self.num_bundles, self.bundle_dim,
                                 -1)  # works since in_channel divisible by bundle dim

        # compute the orthogonal maps
        if self.num_gnn_layers > 0:
            node_rep = self.struct_enc(x, edge_ind)  # edge_ind is without the self loops
        else:
            node_rep = self.struct_enc(x)

        node_rep = node_rep.reshape(num_nodes * self.num_bundles, self.num_bun_params // self.num_bundles)
        node_rep = self.orthogonal(node_rep)
        node_rep = node_rep.reshape(num_nodes, self.num_bundles,
                                    self.bundle_dim, self.bundle_dim)  # want it to be one matrix per channel per node

        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, self.in_channels)

        curr_pow = h
        for k in range(1, self.max_deg+1):
            curr_pow = -self.tau * 1/k * (curr_pow - self.propagate(x=curr_pow,
                                                                    edge_index=edge_index,
                                                                    edge_weight=edge_weight))  # -tau/k*(Id - A)
            h = h + curr_pow

        # apply linear transformation
        h = self.lin(h)

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, self.num_bundles, self.bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.in_channels)
        return h

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class BuNNLayer(nn.Module):
    """ First, BundleConv implements the message passing/diffusion.
    Then AffineUpdate implements the linear map for updating the node features.
    Then BatchNorm implements batch norm.
    Then activation is also implemented.
    Then dropout"""
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()

        # patching for LRGB
        # in_channels = config.dim_in
        # out_channels = config.dim_out
        in_channels = dim_in
        out_channels = dim_out
        bundle_dim = cfg.bundle.bundle_dim
        add_self_loops = True
        normalize = True
        # residual = False
        # num_bundles = in_channels // bundle_dim
        num_bundles = cfg.bundle.num_bundle
        self.dropout = dropout

        self.max_deg = cfg.bundle.max_deg
        self.tau = cfg.bundle.tau
        self.gnn_dim = cfg.bundle.gnn_dim
        self.num_gnn = cfg.bundle.num_gnn
        self.orth_method = cfg.bundle.orth_method

        # self.norm = get_norm(norm)(in_channels)
        self.residual = residual
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.bundle_dim = bundle_dim

        if in_channels != out_channels:
            self.residual = False
            warnings.warn("in and out dimensions do not match, so residual is turned off")

        self.conv = BuNNConv(in_channels=in_channels,
                              bundle_dim=bundle_dim,
                              normalize=normalize,
                              add_self_loops=add_self_loops,
                              num_bundles=num_bundles,
                              num_gnn=cfg.bundle.num_gnn,
                              gnn_dim=cfg.bundle.gnn_dim,
                              dropout=self.dropout,
                              max_deg=self.max_deg,
                              tau=self.tau,
                              orth_meth=self.orth_method)

        if cfg.bundle.batchnorm:
            self.batchnorm = nn.BatchNorm1d(in_channels)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        h = self.conv(x, edge_index)
        if self.residual:
            h = x + h
        h = F.dropout(h, p=self.dropout, training=self.training)

        if cfg.bundle.batchnorm:
            h = self.batchnorm(h)

        batch.x = h
        return batch


register_layer('bunnconv', BuNNLayer)


class LimConv(MessagePassing):
    """ This implements only the message passing part of the layer: it has no parameters any more.
    Note: it does not have parameters in order to have a time parameter in the Layer, which should't applying the update function over and over again"""
    def __init__(self, in_channels: int,
                 out_channels: int,
                 bundle_dim: int,
                 num_bundles: int,
                 num_gnn: int):
        super().__init__()

        assert in_channels % bundle_dim == 0 and out_channels % bundle_dim == 0

        self.in_channels = in_channels
        self.bundle_dim = bundle_dim
        self.num_bundles = num_bundles
        self.num_gnn_layers = num_gnn

        self.lin = torch.nn.Linear(in_channels, out_channels)

        # Neural network used to compute the bundle: either a GNN, or an MLP () if num_gnn_layers is set to 0

        self.struct_enc = Module()

        if self.num_gnn_layers > 0:
            self.struct_enc = SumGNN(in_channels, num_bundles, self.num_gnn_layers)
        else:
            self.struct_enc = torch.nn.Sequential(torch.nn.Linear(in_channels, 2*num_bundles),
                                                    torch.nn.GELU(),
                                                    torch.nn.Linear(2*num_bundles, self.num_bundles),
                                                  )
        assert self.num_bundles % 2 == 0
        assert self.bundle_dim == 2  # currently assumed in the code Must change the dim out in struct enc, and potentially the euler to housholder below

        self.orthogonal = Orthogonal(d=bundle_dim, orthogonal_map="euler")


    def forward(self,  x, edge_index, batch=None):
        """
        x has shape [num_nodes, in_channels]
        edge_index has shape [2, num_edges]
        node_rep has shape [num_nodes, out_channels//bundle_dim, bundle_dim, bundle_dim]
                 and consists of in_channels//bundle_dim orthogonal matrices per node
        """

        num_nodes = x.shape[0]
        vector_field = x.reshape(num_nodes, self.num_bundles, self.bundle_dim, -1)  # works since in_channel divisible by bundle dim

        if self.num_gnn_layers > 0:
            node_rep = self.struct_enc(x, edge_index)
        else:
            node_rep = self.struct_enc(x)
        # node_rep = F.dropout(node_rep, p=self.dropout, training=self.training)
        node_rep = node_rep.reshape(num_nodes * int(self.num_bundles / 2), 2)
        node_rep = torch.tanh(node_rep)
        node_rep = self.orthogonal(node_rep)
        node_rep = node_rep.reshape(num_nodes, self.num_bundles,
                                    self.bundle_dim, self.bundle_dim)  # want it to be one matrix per channel per node


        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, self.in_channels)

        # apply linear transformation
        h = self.lin(h)
        h = global_mean_pool(h, batch)  # analytic form of the limit (without the degree to simplify)
        h = h[batch]  # broadcast back

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, self.num_bundles, self.bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3), vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.in_channels)
        return h

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class LimLayer(nn.Module):
    """ First, BundleConv implements the message passing/diffusion.
    Then AffineUpdate implements the linear map for updating the node features.
    Then BatchNorm implements batch norm.
    Then activation is also implemented.
    Then dropout"""
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()

        # patching for LRGB
        # in_channels = config.dim_in
        # out_channels = config.dim_out
        in_channels = dim_in
        out_channels = dim_out
        bundle_dim = cfg.bundle.bundle_dim
        add_self_loops = True
        normalize = True
        # residual = False
        # num_bundles = in_channels // bundle_dim
        num_bundles = cfg.bundle.num_bundle
        self.dropout = dropout

        # self.norm = get_norm(norm)(in_channels)
        self.residual = residual
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.num_gnn_layers = cfg.bundle.num_gnn


        self.bundle_dim = bundle_dim

        if in_channels != out_channels:
            self.residual = False
            warnings.warn("in and out dimensions do not match, so residual is turned off")

        self.conv = LimConv(in_channels=in_channels,
                            out_channels=out_channels,
                            bundle_dim=bundle_dim,
                            num_bundles=num_bundles,
                            num_gnn=self.num_gnn_layers)

        if cfg.bundle.batchnorm:
            self.batchnorm = nn.BatchNorm1d(in_channels)

    def forward(self, batch):
        x, edge_index, btch = batch.x, batch.edge_index, batch.batch

        h = self.conv(x, edge_index, btch)
        if self.residual:
            h = x + h
        h = F.dropout(h, p=self.dropout, training=self.training)

        if cfg.bundle.batchnorm:
            h = self.batchnorm(h)

        batch.x = h
        return batch

register_layer('limconv', LimLayer)


class BuNNHopConv(MessagePassing):
    """ This implements only the message passing part of the layer: it has no parameters any more.
    Note: it does not have parameters in order to have a time parameter in the Layer, which should't applying the update function over and over again"""
    def __init__(self, in_channels: int,
                 bundle_dim: int,
                 num_bundles: int,
                 normalize: bool = True,
                 add_self_loops=True,
                 dropout=0.0):
        super().__init__()

        assert in_channels % bundle_dim == 0

        self.in_channels = in_channels
        self.bundle_dim = bundle_dim
        self.num_bundles = num_bundles

        self.dropout = dropout

        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.lin = torch.nn.Linear(in_channels, in_channels)

        self.time_steps = cfg.bundle.time_steps
        self.time = self.time_steps[-1]
        self.num_att = len(self.time_steps)

        self.num_gnn_layers = cfg.bundle.num_gnn

        self.attention = torch.nn.Parameter(torch.rand([self.num_att, self.num_bundles]))
        self.register_parameter("attention", self.attention)

        # Neural network used to compute the bundle: either a GNN, or an MLP () if num_gnn_layers is set to 0
        self.struct_enc = Module()
        if self.num_gnn_layers > 0:
            self.struct_enc = SumGNN(in_channels, num_bundles, self.num_gnn_layers)
        else:
            self.struct_enc = torch.nn.Sequential(torch.nn.Linear(in_channels, 2*num_bundles),
                                                    torch.nn.GELU(),
                                                    torch.nn.Linear(2*num_bundles, self.num_bundles),
                                                  )
        assert self.num_bundles % 2 == 0
        assert self.bundle_dim == 2  # currently assumed in the code Must change the dim out in struct enc, and potentially the euler to housholder below

        self.orthogonal = Orthogonal(d=bundle_dim, orthogonal_map="euler")

    def forward(self,  x, edge_index):
        """
        x has shape [num_nodes, in_channels]
        edge_index has shape [2, num_edges]
        node_rep has shape [num_nodes, out_channels//bundle_dim, bundle_dim, bundle_dim]
                 and consists of in_channels//bundle_dim orthogonal matrices per node
        """

        edge_ind = edge_index
        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # adds self loops
                    edge_index, edge_weight=None,
                    add_self_loops=self.add_self_loops, dtype=x[0].dtype)

                deg = torch_geometric.utils.degree(edge_index[0], dtype=edge_index.dtype)
                deg_inv = deg.float().pow(-1)
                col, row = edge_index[0], edge_index[1]
                edge_weight = deg_inv[row]  # degree normalization instead of sqrt from GCN/symmetric normalization
            else:
                raise NotImplementedError
        else:
            edge_weight = None

        num_nodes = x.shape[0]
        vector_field = x.reshape(num_nodes, self.num_bundles, self.bundle_dim, -1)  # works since in_channel divisible by bundle dim

        # compute the orthogonal maps
        if self.num_gnn_layers > 0:
            node_rep = self.struct_enc(x, edge_ind) # edge_ind is without the self loops
        else:
            node_rep = self.struct_enc(x)

        node_rep = F.dropout(node_rep, p=self.dropout, training=self.training)

        node_rep = node_rep.reshape(num_nodes*int(self.num_bundles/2), 2)
        node_rep = torch.tanh(node_rep)
        node_rep = self.orthogonal(node_rep)
        node_rep = node_rep.reshape(num_nodes, self.num_bundles,
                                    self.bundle_dim, self.bundle_dim)  # want it to be one matrix per channel per node

        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, self.in_channels)

        # apply linear transformation
        h = self.lin(h)
        different_times = []
        for t in range(1, self.time+1):
            h = self.propagate(x=h,
                               edge_index=edge_index,
                               edge_weight=edge_weight)
            if t in self.time_steps:
                different_times += [h]
        different_times = torch.stack(different_times)
        attention = self.attention.softmax(dim=0)
        different_times = different_times.reshape([self.num_att, num_nodes, self.num_bundles, -1])
        h = torch.einsum('ad, acdb -> cdb', attention, different_times)
        h = h.reshape([num_nodes, self.in_channels])

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, self.num_bundles, self.bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3), vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.in_channels)
        return h

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class BuNNHopLayer(nn.Module):
    """ First, BundleConv implements the message passing/diffusion.
    Then AffineUpdate implements the linear map for updating the node features.
    Then BatchNorm implements batch norm.
    Then activation is also implemented.
    Then dropout"""
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()

        # patching for LRGB
        # in_channels = config.dim_in
        # out_channels = config.dim_out
        in_channels = dim_in
        out_channels = dim_out
        bundle_dim = cfg.bundle.bundle_dim
        add_self_loops = True
        normalize = True
        # residual = False
        # num_bundles = in_channels // bundle_dim
        num_bundles = cfg.bundle.num_bundle
        self.dropout = dropout

        # self.norm = get_norm(norm)(in_channels)
        self.residual = residual
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.bundle_dim = bundle_dim

        if in_channels != out_channels:
            self.residual = False
            warnings.warn("in and out dimensions do not match, so residual is turned off")

        self.conv = BuNNHopConv(in_channels=in_channels,
                                  bundle_dim=bundle_dim,
                                  normalize=normalize,
                                  add_self_loops=add_self_loops,
                                  num_bundles=num_bundles,
                                  dropout=self.dropout)

        if cfg.bundle.batchnorm:
            self.batchnorm = nn.BatchNorm1d(in_channels)

    def make_weightless(self):
        self.update_fct.make_weightless()
        # if self.norm is not None and not identi:
        #     self.norm.reset_parameters()

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        h = self.conv(x, edge_index)
        if self.residual:
            h = x + h
        h = F.dropout(h, p=self.dropout, training=self.training)


        if cfg.bundle.batchnorm:
            h = self.batchnorm(h)

        batch.x = h
        return batch


register_layer('bunnhopconv', BuNNHopLayer)


class SpecBuNNConv(nn.Module):
    def __init__(self, in_channels: int,
                bundle_dim: int,
                num_bundles: int,
                num_gnn: int,
                gnn_dim: int,
                normalize: bool = True,
                dropout=0.0,
                tau=1,
                orth_meth="householder",
                tau_method="fixed",
                multiscale=False):
        super().__init__()

        assert in_channels % bundle_dim == 0
        
        self.in_channels = in_channels
        self.bundle_dim = bundle_dim
        self.num_bundles = num_bundles
        self.num_bun_params = int((self.bundle_dim-1) * self.bundle_dim / 2 * self.num_bundles)
        self.gnn_dim = gnn_dim

        self.tau_method = tau_method

        self.dropout = dropout

        self.normalize = normalize

        self.lin = torch.nn.Linear(in_channels, in_channels)

        self.tau = tau
        self.multiscale = multiscale
        if self.multiscale:
            half = self.num_bundles//2
            self.taus = nn.Parameter(torch.tensor([10**(k-half)for k in range(self.num_bundles)], dtype=torch.float))  # ensures that tau=1 is in the scales
        else:
            self.taus = nn.Parameter(torch.tensor([self.tau for k in range(self.num_bundles)], dtype=torch.float))
        self.orth_meth = orth_meth

        self.num_gnn_layers = num_gnn # cfg.bundle.num_gnn

        # Neural network used to compute the bundle: either a GNN, or an MLP () if num_gnn_layers is set to 0
        self.struct_enc = Module()
        if self.num_gnn_layers > 0:
            self.struct_enc = SumGNN(dim_in=in_channels, hidden_dim=self.gnn_dim, dim_out=self.num_bun_params, num_layers=self.num_gnn_layers)
        else:
            self.struct_enc = torch.nn.Sequential(torch.nn.Linear(in_channels, self.gnn_dim),
                                                    torch.nn.GELU(),
                                                    torch.nn.Linear(self.gnn_dim, self.num_bun_params),
                                                  )

        self.orthogonal = Orthogonal(d=bundle_dim, orthogonal_map=self.orth_meth)
        
        if cfg.bundle.batchnorm:
            self.batchnorm = nn.BatchNorm1d(self.num_bun_params)

    
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        
        num_nodes = x.shape[0]
        vector_field = x.reshape(num_nodes, self.num_bundles, self.bundle_dim,
                                 -1)  # works since in_channel divisible by bundle dim

        # compute the orthogonal maps
        if self.num_gnn_layers > 0:
            node_rep = self.struct_enc(x, edge_index)  # edge_ind is without the self loops
        else:
            node_rep = self.struct_enc(x)
        if cfg.bundle.batchnorm:
            node_rep = self.batchnorm(node_rep)
        node_rep = node_rep.reshape(num_nodes * self.num_bundles, self.num_bun_params // self.num_bundles)
        node_rep = self.orthogonal(node_rep)
        node_rep = node_rep.reshape(num_nodes, self.num_bundles,
                                    self.bundle_dim, self.bundle_dim)  # want it to be one matrix per channel per node

        
        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, self.in_channels)
        # print("batch shape: ", batch.batch.shape)
        # print("eigval shape: ", batch.eigval.shape)
        # print("og shape: ", h.shape)
        # now message diffusion, we need dense bacth to use the eigenvectors:
        h, mask = to_dense_batch(h, batch.batch)  # [n_graph, max_n_nodes, in_channels]
        # print(h.shape)
        # print("og eigvec: ", batch.eigvec.shape)
        eigvec, mask2 = to_dense_batch(batch.eigvec, batch.batch)  # [n_graphs, max_n_nodes, 500]
        # print("eigvec: ", eigvec.shape)
        eigval = batch.eigval  # [n_graphs, 500]
        
        h = torch.einsum('gvn, gnd -> gvd', eigvec.transpose(1, 2), h)  # project in spectral domain
        fact = torch.exp(-torch.einsum('gv, b -> gvb', eigval, self.taus))  # compute the exp(-t*eigenvalues) to scale each spectral component
        h = h.reshape(h.shape[0], h.shape[1], self.num_bundles, self.bundle_dim, -1)  # reshape to scale each bundle differently
        h = torch.einsum('gvbdc, gvb -> gvbdc', h, fact)  # scale the spectral components
        h = h.reshape(h.shape[0], h.shape[1], -1)  # reshape to previous shape
        h = torch.einsum('gnv, gvd -> gnd', eigvec, h)  #  project back to spatial domain
        h = h[mask]  # remove padding

        # apply linear transformation
        h = self.lin(h)
        
        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, self.num_bundles, self.bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.in_channels)
        
        return h


class SpecBuNNLayer(nn.Module):
    """ First, BundleConv implements the message passing/diffusion.
    Then AffineUpdate implements the linear map for updating the node features.
    Then BatchNorm implements batch norm.
    Then activation is also implemented.
    Then dropout"""
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()

        # patching for LRGB
        # in_channels = config.dim_in
        # out_channels = config.dim_out
        in_channels = dim_in
        out_channels = dim_out
        bundle_dim = cfg.bundle.bundle_dim
        add_self_loops = True
        normalize = True
        # residual = False
        # num_bundles = in_channels // bundle_dim
        self.num_bundles = cfg.bundle.num_bundle
        self.dropout = dropout

        self.max_deg = cfg.bundle.max_deg
        self.tau = cfg.bundle.tau
        self.gnn_dim = cfg.bundle.gnn_dim
        self.num_gnn = cfg.bundle.num_gnn
        self.orth_method = cfg.bundle.orth_method
        
        self.multiscale = cfg.bundle.multiscale

        # self.norm = get_norm(norm)(in_channels)
        self.residual = residual
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.bundle_dim = bundle_dim

        if in_channels != out_channels:
            self.residual = False
            warnings.warn("in and out dimensions do not match, so residual is turned off")

        self.conv = SpecBuNNConv(in_channels=in_channels,
                              bundle_dim=bundle_dim,
                              normalize=normalize,
                              num_bundles=self.num_bundles,
                              num_gnn=cfg.bundle.num_gnn,
                              gnn_dim=cfg.bundle.gnn_dim,
                              dropout=self.dropout,
                              tau=self.tau,
                              orth_meth=self.orth_method,
                              multiscale=self.multiscale)
        
        self.act = register.act_dict[cfg.gnn.act]()

        if cfg.bundle.batchnorm:
            self.batchnorm = nn.BatchNorm1d(in_channels)

    def forward(self, batch):
        h = self.conv(batch)
        h = F.dropout(h, p=self.dropout, training=self.training)
        if cfg.bundle.batchnorm:
            h = self.batchnorm(h)
        h  = self.act(h)
        if self.residual:
            h = batch.x + h
        batch.x = h
        return batch

register_layer('specbunnconv', SpecBuNNLayer)
