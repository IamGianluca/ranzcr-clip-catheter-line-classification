from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision.utils as utils
from PIL import Image
from torch.utils.data import DataLoader


def resize_images_from_folder(in_path: Path, out_path: Path, sz: int):
    image_names = [x.name for x in in_path.iterdir() if x.is_file()]

    pool = Pool()
    routine = partial(resize_image, sz=sz, in_path=in_path, out_path=out_path)
    pool.map(routine, image_names)


def resize_image(image_name: str, sz: int, in_path: Path, out_path: Path):
    img = Image.open(in_path / image_name)
    try:
        resized_img = resize_image_and_pad_if_needed(img=img, sz=sz)
    except ValueError as err:
        print(image_name, img.size, err)
        raise err
    resized_img.save(out_path / image_name)


def resize_image_and_pad_if_needed(img: Image, sz: int):
    """If the desired output image size is larger than the original image,
    place the image in the middle of a black template of the desired size.
    """
    # TODO: check if this works propery for both RGB and black-white images
    # TODO: test bilinear vs other options
    if img.size[0] > sz or img.size[1] > sz:
        resized_img = img.resize((sz, sz))
    else:
        resized_img = Image.new("RGB", (sz, sz))
        resized_img.paste(img, (0, 0))
    return resized_img


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
