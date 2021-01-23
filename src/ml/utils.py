import argparse

import matplotlib.pyplot as plt
import torchvision.utils as utils
from torch.utils.data import DataLoader


def dict_to_args(d: dict):

    args = argparse.Namespace()

    def dict_to_args_recursive(args, d, prefix=""):
        for k, v in d.items():
            if type(v) == dict:
                dict_to_args_recursive(args, v, prefix=k)
            elif type(v) in [tuple, list]:
                continue
            else:
                if prefix:
                    args.__setattr__(prefix + "_" + k, v)
                else:
                    args.__setattr__(k, v)

    dict_to_args_recursive(args, d)
    return args


def plot_a_batch(dl: DataLoader):
    for batch_number, batch in enumerate(dl):
        plt.figure(figsize=(20, 10))
        show_images_in_batch(batch=batch, verbose=False)
        plt.axis("off")
        plt.ioff()
        plt.show()
        if batch_number == 2:
            break


def show_images_in_batch(batch, verbose=False):
    try:
        images, targets = batch
    except ValueError:
        images = batch
        targets = None

    if verbose:
        print(images.shape)
        print(targets)

    grid = utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
