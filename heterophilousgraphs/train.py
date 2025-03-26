import argparse
from tqdm import tqdm
from distutils.util import strtobool
import torch.nn.functional as F

import torch
from torch.cuda.amp import autocast, GradScaler

from model import Model, BUNNModel
# from nsd.models.disc_models import DiscreteBundleSheafDiffusion

from datasets import Dataset
from utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup

from torch_geometric.utils.convert import from_dgl
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from numpy.linalg import eigh


import wandb


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='roman-empire',
                        choices=['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                                 'squirrel', 'squirrel-directed', 'squirrel-filtered', 'squirrel-filtered-directed',
                                 'chameleon', 'chameleon-directed', 'chameleon-filtered', 'chameleon-filtered-directed',
                                 'actor', 'texas', 'texas-4-classes', 'cornell', 'wisconsin'])

    parser.add_argument('--wandb', action='store_true')


    # model architecture
    parser.add_argument('--model', type=str, default='GT-sep',
                        choices=['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep', 'BUNN', 'BUNNHOP', 'SHEAF',
                                 'LimBuNN', "GCNBuNN", "SpecBuNN", "SpecBuNN-sep"])
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])

    # BDL params
    parser.add_argument('--num_bundles', type=int, default=32)
    parser.add_argument('--time_steps', type=int, nargs="+", default=1)
    parser.add_argument('--bundle_dim', type=int, default=2)
    parser.add_argument('--num_gnn', type=int, default=5)
    parser.add_argument('--max_degree', type=int, default=32)
    parser.add_argument('--tau', type=float, default=1.5)
    parser.add_argument('--tau_method', type=str, default="range")  # fixed or range
    parser.add_argument('--gnn_type', type=str, default="SAGE")
    parser.add_argument('--gnn_method', type=str, default="diff")  # diff or shared or shared-recomp
    parser.add_argument('--orth_meth', type=str, default="householder")
    parser.add_argument('--max_eigs', type=int, default=500)

    # regularization
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)

    # training parameters
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--lr_tau', type=float, default=1e-2)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')

    # node feature augmentation
    parser.add_argument('--use_sgc_features', default=False, action='store_true')
    parser.add_argument('--use_identity_features', default=False, action='store_true')
    parser.add_argument('--use_adjacency_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_original_features', default=False, action='store_true')

    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')

    # SHEAF PARAMS
    # Model configuration
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--normalised', dest='normalised', type=str2bool, default=True,
                        help="Use a normalised Laplacian")
    parser.add_argument('--deg_normalised', dest='deg_normalised', type=str2bool, default=False,
                        help="Use a a degree-normalised Laplacian")
    parser.add_argument('--linear', dest='linear', type=str2bool, default=False,
                        help="Whether to learn a new Laplacian at each step.")
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--left_weights', dest='left_weights', type=str2bool, default=True,
                        help="Applies left linear layer")
    parser.add_argument('--right_weights', dest='right_weights', type=str2bool, default=True,
                        help="Applies right linear layer")
    parser.add_argument('--add_lp', dest='add_lp', type=str2bool, default=False,
                        help="Adds fixed high pass filter in the restriction maps")
    parser.add_argument('--add_hp', dest='add_hp', type=str2bool, default=False,
                        help="Adds fixed low pass filter in the restriction maps")
    parser.add_argument('--use_act', dest='use_act', type=str2bool, default=True)
    parser.add_argument('--second_linear', dest='second_linear', type=str2bool, default=False)
    parser.add_argument('--orth', type=str, choices=['matrix_exp', 'cayley', 'householder', 'euler'],
                        default='householder', help="Parametrisation to use for the orthogonal group.")
    parser.add_argument('--sheaf_act', type=str, default="tanh", help="Activation to use in sheaf learner.")
    parser.add_argument('--edge_weights', dest='edge_weights', type=str2bool, default=True,
                        help="Learn edge weights for connection Laplacian")
    parser.add_argument('--sparse_learner', dest='sparse_learner', type=str2bool, default=False)

    parser.add_argument('--max_t', type=float, default=1.0, help="Maximum integration time.")




    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    print(args)
    return args


def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')

def train_step(model, dataset, optimizer, scheduler, scaler, amp=False, sheaf_model=False):
    model.train()

    with autocast(enabled=amp):
        if sheaf_model:
            logits = model(x=dataset.node_features)  # the graph is used to build the model
            if logits.shape[-1] == 1:
                logits = logits.squeeze()
        else:
            logits = model(graph=dataset.graph, x=dataset.node_features, eig_vecs=dataset.eig_vecs, eig_vals=dataset.eig_vals)
        loss = dataset.loss_fn(input=logits[dataset.train_idx], target=dataset.labels[dataset.train_idx])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def evaluate(model, dataset, amp=False, sheaf_model=False):
    model.eval()

    with autocast(enabled=amp):
        if sheaf_model:
            logits = model(x=dataset.node_features)  # the graph is used to build the model
        else:
            logits = model(graph=dataset.graph, x=dataset.node_features, eig_vecs=dataset.eig_vecs, eig_vals=dataset.eig_vals)
    metrics = dataset.compute_metrics(logits)

    return metrics


def main():
    args = get_args()

    # Set the seed for everything
    torch.manual_seed(213982)
    torch.cuda.manual_seed(213982)
    torch.cuda.manual_seed_all(213982)
    file_name = "saved_models"

    if args.wandb:
        wdb_run = wandb.init(project="bundles-het",
                         group=args.dataset + "meta",
                         name=args.name)
        wdb_run.config.update(args)

    dataset = Dataset(name=args.dataset,
                      add_self_loops=(args.model in ['GCN', 'GAT', 'GT']),
                      device=args.device,
                      use_sgc_features=args.use_sgc_features,
                      use_identity_features=args.use_identity_features,
                      use_adjacency_features=args.use_adjacency_features,
                      do_not_use_original_features=args.do_not_use_original_features)

    logger = Logger(args, metric=dataset.metric, num_data_splits=dataset.num_data_splits)

    for run in range(1, args.num_runs + 1):
        sheaf_model = False
        if args.model in ["BUNN", "BUNNHOP", "LimBuNN", "GCNBuNN", "SpecBuNN", "SpecBuNN-sep"]:
            model = BUNNModel(model_name=args.model,
                              num_layers=args.num_layers,
                              input_dim=dataset.num_node_features,
                              hidden_dim=args.hidden_dim,
                              bundle_dim=args.bundle_dim,
                              output_dim=dataset.num_targets,
                              hidden_dim_multiplier=args.hidden_dim_multiplier,
                              num_heads=args.num_heads,
                              normalization=args.normalization,
                              dropout=args.dropout,
                              num_bundles=args.num_bundles,
                              gnn_type=args.gnn_type,
                              gnn_method=args.gnn_method,
                              num_gnn=args.num_gnn,
                              tau=args.tau,
                              tau_method=args.tau_method,
                              max_degree=args.max_degree,
                              orth_meth=args.orth_meth,
                              max_eigs=args.max_eigs
                             )
        elif args.model == 'SHEAF':
            edge_index = from_dgl(dataset.graph).edge_index.long()  # device?
            args.graph_size = dataset.node_features.size(0)
            args.input_dim = dataset.num_node_features
            args.output_dim = dataset.num_targets

            args.layers = args.num_layers
            args.hidden_channels = args.hidden_dim

            model = DiscreteBundleSheafDiffusion(edge_index, vars(args))
            sheaf_model = True
        else:
            model = Model(model_name=args.model,
                          num_layers=args.num_layers,
                          input_dim=dataset.num_node_features,
                          hidden_dim=args.hidden_dim,
                          output_dim=dataset.num_targets,
                          hidden_dim_multiplier=args.hidden_dim_multiplier,
                          num_heads=args.num_heads,
                          normalization=args.normalization,
                          dropout=args.dropout)

        model.to(args.device)

        if args.model == "SHEAF":
            sheaf_learner_params, other_params = model.grouped_parameters()
            optimizer = torch.optim.Adam([
                {'params': sheaf_learner_params, 'weight_decay': args.weight_decay},
                {'params': other_params, 'weight_decay': args.weight_decay}
            ], lr=args.lr)

        elif args.model in ['SpecBuNN', 'SpecBuNN-sep']:
            w_params, bunn_params, tau_params = model.grouped_parameters()
            optimizer = torch.optim.Adam([
                {'params': w_params, 'weight_decay': args.weight_decay, 'lr': args.lr},
                {'params': bunn_params, 'weight_decay': args.weight_decay, 'lr': args.lr},
                {'params': tau_params, 'weight_decay': args.weight_decay, 'lr': args.lr_tau}
            ])
        else:
            parameter_groups = get_parameter_groups(model)
            optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)

        scaler = GradScaler(enabled=args.amp)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                                 num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)

        logger.start_run(run=run, data_split=dataset.cur_data_split + 1)
        with tqdm(total=args.num_steps, desc=f'Run {run}', disable=args.verbose) as progress_bar:
            for step in range(1, args.num_steps + 1):
                train_step(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, amp=args.amp, sheaf_model=sheaf_model)
                metrics = evaluate(model=model, dataset=dataset, amp=args.amp, sheaf_model=sheaf_model)
                logger.update_metrics(metrics=metrics, step=step)

                progress_bar.update()
                progress_bar.set_postfix({metric: f'{value:.2f}' for metric, value in metrics.items()})
        logger.finish_run()
        model.cpu()
        dataset.next_data_split()

        # torch.save(model, file_name+"/"+str("GCNBuNN") +str(args.dataset)+"-"+str(run))

    logger.print_metrics_summary()
    metrics = logger.get_metric_summary()
    if args.wandb:
        wdb_run.log(metrics)
        wdb_run.finish()

if __name__ == '__main__':
    main()
