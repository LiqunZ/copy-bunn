from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('bundle')
def set_cfg_bundle(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.bundle = CN()
    cfg.bundle.bundle_dim = 2

    cfg.bundle.time_steps = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 64]
    cfg.bundle.num_bundle = 16

    cfg.bundle.max_deg = 32
    cfg.bundle.tau = 1.0

    cfg.bundle.num_gnn = 3
    cfg.bundle.gnn_dim = 16

    cfg.bundle.orth_method = "householder"
    cfg.bundle.tau_method = "fixed"

    cfg.bundle.batchnorm = True
    cfg.bundle.multiscale = False