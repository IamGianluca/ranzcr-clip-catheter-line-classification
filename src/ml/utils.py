import matplotlib.pyplot as plt
import torchvision.utils as utils
from torch.utils.data import DataLoader


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
