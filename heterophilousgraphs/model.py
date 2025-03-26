import torch
from torch import nn
from modules import (ResidualModuleWrapper, FeedForwardModule, GCNModule, SAGEModule, GATModule, GATSepModule,
                     TransformerAttentionModule, TransformerAttentionSepModule, SumGNNModule)
from bunn_modules import BUNNModule, BUNNHOPModule, LimBuNNModule, GCNBuNNModule, SpecBUNNModule, SpecBUNNSepModule
from bundles.orthogonal import Orthogonal
from torch.nn import Module


MODULES = {
    'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
    'SAGE': [SAGEModule],
    'GAT': [GATModule],
    'GAT-sep': [GATSepModule],
    'GT': [TransformerAttentionModule, FeedForwardModule],
    'GT-sep': [TransformerAttentionSepModule, FeedForwardModule],
    'BUNN': [BUNNModule],
    'BUNNHOP': [BUNNHOPModule],
    'SumGNN': [SumGNNModule],
    'LimBuNN': [LimBuNNModule],
    'GCNBuNN': [GCNBuNNModule],
    'SpecBuNN': [SpecBUNNModule],
    'SpecBuNN-sep': [SpecBUNNSepModule]}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


class Model(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout):

        super().__init__()

        normalization = NORMALIZATION[normalization]

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_val = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x, **kwargs):
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class BUNNModel(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, bundle_dim, output_dim,
                 hidden_dim_multiplier, num_heads,
                 normalization, dropout, num_bundles, num_gnn,
                 tau=0.1, tau_method="fixed", max_degree=12,
                 gnn_type="SAGE", gnn_method="diff",
                 orth_meth="householder", learn_tau=True, max_eigs=500):

        super().__init__()

        self.normalization_st = normalization
        normalization = NORMALIZATION[normalization]

        self.hidden_dim = hidden_dim
        self.bundle_dim = bundle_dim
        if num_bundles is None:
            num_bundles = (self.hidden_dim // self.bundle_dim)
        self.num_bundles = int(num_bundles)
        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_val = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.tau_method = tau_method
        self.gnn_method = gnn_method

        self.residual_modules = nn.ModuleList()
        self.orth_meth = orth_meth
        self.orthogonal = Orthogonal(d=bundle_dim, orthogonal_map=self.orth_meth)
        self.num_gnn = num_gnn

        self.num_bun_params = int((self.bundle_dim-1) * self.bundle_dim / 2 * self.num_bundles)

        self.enc_computers = nn.ModuleList()

        for _ in range(num_layers):
            if self.gnn_method == "diff":
                if self.num_gnn > 0:
                    self.enc_computers.append(
                        Model(gnn_type, num_gnn,  # TODO: add this as hyperparam
                              hidden_dim,
                              hidden_dim=hidden_dim,
                              output_dim=self.num_bun_params,
                              hidden_dim_multiplier=hidden_dim_multiplier,
                              normalization=self.normalization_st,
                              dropout=self.dropout_val,
                              num_heads=0,
                              )
                    )
                else:
                    self.enc_computers.append(
                        FeedForwardModule(dim=self.num_bun_params, input_dim_multiplier=hidden_dim // self.num_bun_params,
                                          hidden_dim_multiplier=hidden_dim // self.num_bun_params, dropout=dropout)
                    )
            elif self.gnn_method == "shared" or self.gnn_method == "shared-recomp":
                self.enc_computers.append(
                    FeedForwardModule(dim=self.num_bun_params, input_dim_multiplier=hidden_dim // self.num_bun_params,
                                      hidden_dim_multiplier=hidden_dim // self.num_bun_params, dropout=dropout)
                    )
            else:
                raise NotImplementedError

            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        bundle_dim=bundle_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout,
                                                        num_bundles=num_bundles,
                                                        tau=tau,
                                                        tau_method=self.tau_method,
                                                        max_degree=max_degree,
                                                        learn_tau=learn_tau,
                                                        max_eigs=max_eigs
                                                        )
                self.residual_modules.append(residual_module)

            if self.gnn_method == "shared" or self.gnn_method == "shared-recomp":
                if self.num_gnn > 0:
                    self.shared_gnn = Model(gnn_type, num_gnn,  # TODO: add this as hyperparam
                                            hidden_dim,
                                            hidden_dim=hidden_dim,
                                            output_dim=hidden_dim,
                                            hidden_dim_multiplier=hidden_dim_multiplier,
                                            normalization=self.normalization_st,
                                            dropout=self.dropout_val,
                                            num_heads=0,
                                            )
                else:
                    self.shared_gnn.append(
                        FeedForwardModule(dim=self.num_bun_params,
                                          input_dim_multiplier=hidden_dim // self.num_bun_params,
                                          hidden_dim_multiplier=hidden_dim // self.num_bun_params, dropout=dropout)
                    )

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x, eig_vecs=None, eig_vals=None):
        num_nodes = x.shape[0]
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        if self.gnn_method == "shared":
            shared_emb = self.shared_gnn(graph, x)

        for k, residual_module in enumerate(self.residual_modules):
            if self.gnn_method == "shared":
                node_rep = self.enc_computers[k](graph, shared_emb)
            elif self.gnn_method == "shared-recomp":
                shared_emb = self.shared_gnn(graph, x)  # recompute
                node_rep = self.enc_computers[k](graph, shared_emb)
            else:
                node_rep = self.enc_computers[k](graph, x)

            node_rep = node_rep.reshape(num_nodes * self.num_bundles, self.num_bun_params // self.num_bundles)
            node_rep = self.orthogonal(node_rep)
            # node_rep = torch.rand([num_nodes * self.num_bundles, self.bundle_dim, self.bundle_dim], device=x.device) # to debug...
            node_rep = node_rep.reshape(num_nodes, self.num_bundles,
                                        self.bundle_dim, self.bundle_dim)  # want it to be one matrix per channel per node
            x = residual_module(graph, x, node_rep, eig_vecs=eig_vecs, eig_vals=eig_vals)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x

    def grouped_parameters(self):
        w_params, bunn_params, tau_params = [], [], []
        for name, param in self.named_parameters():
            if "tau" in name:
                tau_params.append(param)
            elif "enc" in name:
                bunn_params.append(param)
            else:
                w_params.append(param)
        return w_params, bunn_params, tau_params

