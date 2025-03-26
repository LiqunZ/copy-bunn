"""
Code is adapted from https://github.com/twitter-research/cwn/blob/main/data/data_loading.py
Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
Copyright (c) 2021 The CWN Project Authors
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
from math import pi

from definitions import ROOT_DIR
from synthexp.data.datasets.zinc import load_zinc_dataset
from synthexp.data.utils.dist_transform import (DistTransform, generate_dist_data, generate_curved_line_data,
                                                         get_tile, get_many_tile)
from .utils.barbell import gen_many_barbells
from .utils.clique import gen_many_cliques


def load_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'),
                 pre_transform=None, transform=None, config=None):
    """Returns an AugmentedDataset with the specified name and initialised with the given params."""

    train_data, val_data, test_data = _lookup_dataset(name, root, pre_transform, transform, config)

    return train_data, val_data, test_data


def _lookup_dataset(name, root, pre_transform, transform, config=None):
    """ helper function to lookup and load the datasets"""
    if name == 'ZINC-distance':
        # compute the synthetic zinc, and prunes the dataset and generates the samples on the spot
        # example of name: 'ZINC-pow_1_nbg_100_samples_100', last digits determine number of samples,
        # and before that determines number of graphs in the dataset
        pow = config.data.pow
        nb_gr = config.data.nb_graphs
        nb_samples = config.data.samples
        train_data, val_data, test_data = load_zinc_dataset(root, name,
                                                              pre_transform=DistTransform(deg=pow).transform,
                                                              transform=transform)
        train_data_shuffled = train_data.shuffle()[:nb_gr]
        train_data = generate_dist_data(train_data_shuffled, num_gen=nb_samples)
        val_data = generate_dist_data(train_data_shuffled, num_gen=max(nb_samples//5, 1)) # use same graphs but different samplings
        test_data = generate_dist_data(train_data_shuffled, num_gen=max(nb_samples//5, 1))
        return train_data, val_data, test_data
    elif name == "sheaf_learning":
        train_data = generate_curved_line_data(num_samples=config.data.samples, theta=config.data.theta*pi, unique_id=config.data.unique_id)
        val_data = generate_curved_line_data(num_samples=max(config.data.samples//5, 1), theta=config.data.theta*pi, unique_id=config.data.unique_id)
        test_data = generate_curved_line_data(num_samples=max(config.data.samples//5, 1), theta=config.data.theta*pi, unique_id=config.data.unique_id)
        return train_data, val_data,  test_data
    elif name == "unif_expr":
        train_data, val_data, test_data = (get_many_tile(num_samples=config.data.samples, tile=config.data.num_nodes),
                                           get_many_tile(num_samples=max(config.data.samples//5, 1), tile=config.data.num_nodes),
                                           get_tile(num_samples=max(config.data.samples//5, 1), tile=config.data.num_nodes + 10))
        return train_data, val_data, test_data
    elif name == "barbell":
        train_data, val_data, test_data = (gen_many_barbells(num_samples=config.data.samples, n=config.data.num_nodes),
                                           gen_many_barbells(num_samples=config.data.samples, n=config.data.num_nodes),
                                           gen_many_barbells(num_samples=config.data.samples, n=config.data.num_nodes))
        return train_data, val_data, test_data
    elif name == "clique":
        train_data, val_data, test_data = (gen_many_cliques(num_samples=config.data.samples, n=config.data.num_nodes),
                                           gen_many_cliques(num_samples=config.data.samples, n=config.data.num_nodes),
                                           gen_many_cliques(num_samples=config.data.samples, n=config.data.num_nodes))
        return train_data, val_data, test_data
    else:
        raise ValueError("Dataset not found.")
