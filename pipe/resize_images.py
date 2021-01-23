import constants

import ml.vision.utils as utils

DESIRED_IMAGE_SIZES = [128, 192, 256, 384, 512, 768, 1024]


def main():
    for sz in DESIRED_IMAGE_SIZES:
        for folder_name in ["train", "test"]:
            in_path = constants.data_path / folder_name
            out_path = constants.data_path / f"{folder_name}_{sz}"

            if in_path == out_path:
                raise ValueError("in_path and out_path cannot be the same.")

            if not out_path.exists():
                out_path.mkdir()

            print(f"Resizing images in {in_path} to size {sz}x{sz}...")
            utils.resize_images_from_folder(
                in_path=in_path,
                out_path=out_path,
                sz=sz,
            )


if __name__ == "__main__":
    main()
