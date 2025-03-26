from newsynth.nn.model import ModelNode, BuNNNode
from torch_geometric.utils.random import erdos_renyi_graph, barabasi_albert_graph
import torch
from heterophilousgraphs.bundles.orthogonal import Orthogonal


def taylor_approx_test():
    # makes sure that for high enough degrees the taylor approx gives the same output

    edge_index = erdos_renyi_graph(10, 0.6)
    x = torch.rand([10, 3])
    max_deg = 8
    mod = BuNNNode(3, 1, 6, 2, 'TaylorBuNN', num_bundles=2, bundle_dim=3,
                   tau=0.1, max_deg=max_deg, normalize='sym')
    out8 = mod(x, edge_index)

    for conv in mod.convs:
        conv.max_deg = max_deg+1
    out9 = mod(x, edge_index)
    assert torch.allclose(out8, out9, atol=1e-5)
    assert torch.allclose(out8, out9, atol=1e-6)
    assert torch.allclose(out8, out9, atol=1e-7)

def test_spectral_and_taylor():

    num_nodes = 1000
    # edge_index = barabasi_albert_graph(num_nodes, 3)
    edge_index = erdos_renyi_graph(num_nodes, 0.2)
    x = torch.rand([num_nodes, 3]) * 10
    tau = 1
    mod = BuNNNode(3, 1, 6, 3, 'SpectralBuNN', num_bundles=2, bundle_dim=3,
                   tau=tau, normalize='sym', learn_tau=False, act='gelu')

    mod2 = BuNNNode(3, 1, 6, 3, 'TaylorBuNN', num_bundles=2, bundle_dim=3,
                    tau=tau, max_deg=10, normalize='sym', learn_tau=False, act='gelu')
    mod2.enc = mod.enc
    mod2.dec = mod.dec
    for i in range(len(mod.convs)):
        mod2.convs[i].linear = mod.convs[i].linear
        mod2.convs[i].bias = mod.convs[i].bias
        mod2.node_encs[i] = mod.node_encs[i]

    out = mod(x, edge_index)
    out2 = mod2(x, edge_index)
    assert torch.allclose(out, out2, atol=1e-7)

def test_all_runs():
    num_nodes = 1000
    edge_index = erdos_renyi_graph(num_nodes, 0.2)
    x = torch.rand([num_nodes, 3]) * 10
    for layer_type in ['GCN', 'SAGE', 'GIN', 'Sum', 'MLP']:
        mod = ModelNode(3, 1, 6, 3, layer_type, num_bundles=2, bundle_dim=3,
                   tau=1, normalize='sym', learn_tau=False)
        mod(x, edge_index)
    for layer_type in ['TaylorBuNN', 'SpectralBuNN', 'LimBuNN']:
        for gnn_type in ['GCN', 'SAGE', 'GIN', 'Sum', 'MLP']:
            learn_tau = (layer_type == 'SpectralBuNN')
            mod = BuNNNode(3, 1, 6, 3, layer_type, num_bundles=2, bundle_dim=3,
                           tau=1, normalize='sym', learn_tau=learn_tau, gnn_type=gnn_type)
            mod(x, edge_index)
            if layer_type == 'SpectralBuNN':
                mod = BuNNNode(3, 1, 6, 3, layer_type, num_bundles=2, bundle_dim=3,
                               tau=1, normalize='sym', learn_tau=True, gnn_type=gnn_type)
                mod(x, edge_index)

def test_infty():
    mod1 = BuNNNode(3, 1, 6, 2, 'SpectralBuNN', num_bundles=2, bundle_dim=3,
                    normalize='sym')
    mod2 = BuNNNode(3, 1, 6, 2, 'LimBuNN', num_bundles=2, bundle_dim=3,
                    normalize='sym')
    mod2.enc = mod1.enc
    mod2.dec = mod1.dec
    for i in range(len(mod1.convs)):
        mod2.convs[i].linear = mod1.convs[i].linear
        mod2.convs[i].bias = mod1.convs[i].bias
        mod2.node_encs[i] = mod1.node_encs[i]

    edge_index = erdos_renyi_graph(10, 0.6)
    x = torch.rand([10, 3])
    for i in range(len(mod1.convs)):
        mod1.convs[i].tau = torch.ones(mod1.convs[i].tau.shape) * 10000  # time far in advance
    out1, out2 = mod1(x, edge_index), mod2(x, edge_index)

    assert torch.allclose(out1, out2, atol=1e-5)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_orth():
    d = 10
    orth = Orthogonal(d, "householder")
    x = torch.rand([100, int(d * (d - 1) / 2)])
    rot = orth(x)
    ident = torch.einsum('abc, acd -> abd', rot, rot.transpose(1, 2))
    assert torch.allclose(ident, torch.eye(ident.size(1)).unsqueeze(0).repeat(ident.size(0), 1, 1), atol=1e-5)

    d = 2
    orth = Orthogonal(d, "householder")
    x = torch.rand([100, int(d * (d - 1) / 2)])
    rot = orth(x)
    ident = torch.einsum('abc, acd -> abd', rot, rot.transpose(1, 2))
    assert torch.allclose(ident, torch.eye(ident.size(1)).unsqueeze(0).repeat(ident.size(0), 1, 1), atol=1e-5)

    d = 2
    orth = Orthogonal(d, "euler")
    x = torch.rand([100, int(d * (d - 1) / 2)])
    rot = orth(x)
    ident = torch.einsum('abc, acd -> abd', rot, rot.transpose(1, 2))
    assert torch.allclose(ident, torch.eye(ident.size(1)).unsqueeze(0).repeat(ident.size(0), 1, 1), atol=1e-5)
