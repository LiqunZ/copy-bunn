import torch
from torch import nn
from dgl import ops
from modules import FeedForwardModule
from torch.nn import Parameter


class BUNNModule(nn.Module):
    # BuNN mnodule
    def __init__(self, dim, hidden_dim_multiplier, bundle_dim, num_bundles, dropout, max_degree=None, tau=0.1, **kwargs):
        super().__init__()
        self.bundle_dim = bundle_dim
        self.num_bundles = num_bundles
        self.dim = dim

        self.max_degree = max_degree
        self.tau = tau

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)


    def forward(self, graph, x, node_rep, **kwargs):
        num_nodes = x.shape[0]
        num_bundles = node_rep.shape[1]
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        vector_field = x.reshape(num_nodes, num_bundles, self.bundle_dim,
                                 -1)  # works since self.dim divisible by bundle dim

        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, self.dim)
        curr_pow = h
        for k in range(1, self.max_degree+1):
            curr_pow = -self.tau * 1/k * (curr_pow - ops.u_mul_e_sum(graph, curr_pow, norm_coefs))  # -tau/k*(Id - A)
            h = h + curr_pow

        # Importantly, the feedforward is before pulling back
        h = torch.cat([x, h], axis=1)  # SAGE part
        h = self.feed_forward_module(graph, h)

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, self.num_bundles, self.bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.dim)
        return h


class BUNNHOPModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, bundle_dim, num_bundles, dropout, time_steps=None, **kwargs):
        super().__init__()
        self.bundle_dim = bundle_dim
        self.num_bundles = num_bundles
        self.dim = dim
        if time_steps is None:
            self.time_steps = [1, 2, 4, 8, 16]
        else:
            self.time_steps = time_steps
        self.time = self.time_steps[-1]

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

        self.num_att = len(self.time_steps)
        self.attention = torch.nn.Parameter(torch.rand([self.num_att, self.num_bundles]))
        self.register_parameter("attention", self.attention)
        # assert self.time > 20

    def forward(self, graph, x, node_rep, **kwargs):
        num_nodes = x.shape[0]
        num_bundles = node_rep.shape[1]
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        vector_field = x.reshape(num_nodes, num_bundles, self.bundle_dim,
                                 -1)  # works since self.dim divisible by bundle dim

        ## Option 1:
        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)

        h = vector_field.reshape(num_nodes, self.dim)
        different_times = []
        for t in range(1, self.time+1):
            h = ops.u_mul_e_sum(graph, h, norm_coefs)
            if t in self.time_steps:
                different_times += [h]
        different_times = torch.stack(different_times)
        different_times = different_times.reshape([self.num_att, num_nodes, self.num_bundles, -1])

        attention = self.attention.softmax(dim=0)
        h = torch.einsum('ad, acdb -> cdb', attention, different_times)
        h = h.reshape([num_nodes, -1])

        # Importantly, the feedforward is before pulling back
        h = torch.cat([x, h], axis=1)  # SAGE part
        h = self.feed_forward_module(graph, h)

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, self.num_bundles, self.bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.dim)
        return h


class LimBuNNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, bundle_dim, num_bundles, dropout, max_degree=None, tau=0.1,
                 **kwargs):
        super().__init__()
        self.bundle_dim = bundle_dim
        self.num_bundles = num_bundles
        self.dim = dim

        self.max_degree = max_degree
        self.tau = tau

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x, node_rep, **kwargs):
        num_nodes = x.shape[0]
        num_bundles = node_rep.shape[1]
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        vector_field = x.reshape(num_nodes, num_bundles, self.bundle_dim,
                                 -1)  # works since self.dim divisible by bundle dim

        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, self.dim)

        h = h.mean(dim=0).broadcast_to([num_nodes, self.dim])  # replaces the message passing step

        # Importantly, the feedforward is before pulling back
        h = torch.cat([x, h], axis=1)  # SAGE part
        h = self.feed_forward_module(graph, h)

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, self.num_bundles, self.bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.dim)
        return h

class GCNBuNNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, bundle_dim, num_bundles, dropout, max_degree=None, tau=0.1,
                     **kwargs):
            super().__init__()
            self.bundle_dim = bundle_dim
            self.num_bundles = num_bundles
            self.dim = dim

            self.max_degree = max_degree
            self.tau = tau

            self.feed_forward_module = FeedForwardModule(dim=dim,
                                                         input_dim_multiplier=2,
                                                         hidden_dim_multiplier=hidden_dim_multiplier,
                                                         dropout=dropout)

    def forward(self, graph, x, node_rep, **kwargs):
        num_nodes = x.shape[0]
        # num_bundles = node_rep.shape[1]
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        # vector_field = x.reshape(num_nodes, num_bundles, self.bundle_dim,
        #                          -1)  # works since self.dim divisible by bundle dim
        #
        # # first transform into the 'edge space'
        # vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        # h = vector_field.reshape(num_nodes, self.dim)
        h = x
        curr_pow = h
        for k in range(1, self.max_degree + 1):
            curr_pow = -self.tau * 1 / k * (
                        curr_pow - ops.u_mul_e_sum(graph, curr_pow, norm_coefs))  # -tau/k*(Id - A)
            h = h + curr_pow

        # Importantly, the feedforward is before pulling back
        h = torch.cat([x, h], axis=1)  # SAGE part
        h = self.feed_forward_module(graph, h)

        # transform back to 'node space'
        # vector_field = h.reshape(num_nodes, self.num_bundles, self.bundle_dim, -1)
        # vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
        #                             vector_field)  # inverse is transpose
        # h = vector_field.reshape(num_nodes, self.dim)
        return h


class SpecBUNNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, bundle_dim, num_bundles,
                 dropout,  tau=0.1, tau_method="fixed", learn_tau=True, max_eigs=500, **kwargs):
        super().__init__()
        self.bundle_dim = bundle_dim
        self.num_bundles = num_bundles
        self.dim = dim
        self.max_eigs=max_eigs

        self.relu = torch.nn.ReLU()

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=1,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

        if learn_tau:
            if tau_method == "fixed":
                self.tau = Parameter((torch.rand([self.num_bundles]) - 0.5) + tau)  # uniform random centered at tau
            elif tau_method == "range":
                tau_range = torch.arange(1, self.num_bundles+1, dtype=torch.float) / self.num_bundles * tau  # ranges from tau/num_bundles to tau
                self.tau = Parameter(tau_range)
            else:
                raise NotImplementedError
        else:
            big_tau = tau * torch.ones([self.num_bundles], dtype=torch.float)
            self.register_buffer('tau', big_tau)
            self.tau = big_tau


    def forward(self, graph, x, node_rep, eig_vecs=None, eig_vals=None, **kwargs):
        num_nodes = x.shape[0]
        bundle_dim = node_rep.shape[-1]
        num_bundles = node_rep.shape[1]
        num_eigs = eig_vecs.shape[1]

        vector_field = x.reshape(num_nodes, num_bundles, bundle_dim,
                                 -1)  # works since dim divisible by bundle dim
        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, self.dim)

        # compute projections
        eig_vecs_chopped = eig_vecs[:, :self.max_eigs]  # [n, max_eigs]
        proj = torch.einsum('kn, nd -> kd', eig_vecs_chopped.transpose(0, 1), h)  # phi^T*X
        proj = proj.reshape(num_eigs, num_bundles, bundle_dim, -1)
        # compute exponentials
        rep_eigvals = eig_vals.unsqueeze(1).repeat(1, num_bundles)
        exponent = torch.einsum('kb, b -> kb', rep_eigvals, self.relu(self.tau))  # -lambda*tau
        expo = torch.exp(-exponent)
        # evolve the projections
        evol = torch.einsum('kbdc, kb -> kbdc', proj, expo)
        # unproject
        unproj = torch.einsum('nk, kbdc -> nbdc', eig_vecs_chopped, evol)
        h = unproj.reshape(num_nodes, self.dim)

        h = self.feed_forward_module(graph, h)

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, num_bundles, bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.dim)

        return h

class SpecBUNNSepModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, bundle_dim, num_bundles,
                 dropout,  tau=0.1, learn_tau=True, tau_method="fixed", max_eigs=500, **kwargs):
        super().__init__()
        self.bundle_dim = bundle_dim
        self.num_bundles = num_bundles
        self.dim = dim
        self.max_eigs=max_eigs


        self.relu = torch.nn.ReLU()

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

        if learn_tau:
            if tau_method == "fixed":
                self.tau = Parameter((torch.rand([self.num_bundles]) - 0.5) + tau)  # uniform random centered at tau
            elif tau_method == "range":
                tau_range = torch.arange(1, self.num_bundles+1, dtype=torch.float) / self.num_bundles * tau  # ranges from tau/num_bundles to tau
                self.tau = Parameter(tau_range)
            else:
                raise NotImplementedError
        else:
            big_tau = tau * torch.ones([self.num_bundles], dtype=torch.float)
            self.register_buffer('tau', big_tau)
            self.tau = big_tau


    def forward(self, graph, x, node_rep, eig_vecs=None, eig_vals=None, **kwargs):
        num_nodes = x.shape[0]
        bundle_dim = node_rep.shape[-1]
        num_bundles = node_rep.shape[1]
        num_eigs = eig_vecs.shape[1]

        vector_field = x.reshape(num_nodes, num_bundles, bundle_dim,
                                 -1)  # works since dim divisible by bundle dim
        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, self.dim)

        # do the same with x
        vector_field_x = x.reshape(num_nodes, num_bundles, bundle_dim, -1)  # works since dim divisible by bundle dim
        vector_field_x = torch.einsum('abcd, abde -> abce', node_rep, vector_field_x)
        h_x = vector_field_x.reshape(num_nodes, self.dim)

        # compute projections
        eig_vecs_chopped = eig_vecs[:, :self.max_eigs]  # [n, max_eigs]
        proj = torch.einsum('kn, nd -> kd', eig_vecs_chopped.transpose(0, 1), h)  # phi^T*X
        proj = proj.reshape(self.max_eigs, num_bundles, bundle_dim, -1)
        # compute exponentials
        eig_vals_chopped = eig_vals[:self.max_eigs]
        rep_eigvals = eig_vals_chopped.unsqueeze(1).repeat(1, num_bundles)
        exponent = torch.einsum('kb, b -> kb', rep_eigvals, self.relu(self.tau))  # -lambda*tau
        expo = torch.exp(-exponent)
        # evolve the projections
        evol = torch.einsum('kbdc, kb -> kbdc', proj, expo)
        # unproject
        unproj = torch.einsum('nk, kbdc -> nbdc', eig_vecs_chopped, evol)
        h = unproj.reshape(num_nodes, self.dim)

        comb = torch.cat([h_x, h], axis=1)
        h = self.feed_forward_module(graph, comb)

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, num_bundles, bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.dim)

        return h
