from synthexp.nn.model import BuNNNode, ModelNode, ConstantHalf100, ConstantFull
def model_lookup(config):
    if config.model.model_name == 'modelnode':
        model = ModelNode(
            in_dim=config.model.input_dim,
            out_dim=config.model.out_dim,
            hidden_dim=config.model.hidden_dim,
            num_layer=config.model.num_layers,
            layer_type=config.model.layer_type,
            act=config.model.act,
            bias=config.model.bias,
            k=config.model.k,
            heads=config.model.heads
        )
    elif config.model.model_name == 'bunnnode':
        model = BuNNNode(
            in_dim=config.model.input_dim,
            out_dim=config.model.out_dim,
            hidden_dim=config.model.hidden_dim,
            num_layer=config.model.num_layers,
            act=config.model.act,
            bias=config.model.bias,
            layer_type=config.model.layer_type,
            bundle_dim=config.model.bundle_dim,
            num_bundles=config.model.num_bundle,
            tau=config.model.time,
            max_deg=config.model.max_deg,
            num_gnn_layer=config.model.num_gnn_layers,
            gnn_type=config.model.gnn_type,
            learn_tau=config.model.learn_tau
        )
    elif config.model.model_name == "ConstantHalf":
        model = ConstantHalf100(config.data.num_nodes)
    elif config.model.model_name == "ConstantFull":
        model = ConstantFull(config.data.num_nodes)
    else:
        raise ValueError(f'Unknown model {config.model.model_name}')
    return model
