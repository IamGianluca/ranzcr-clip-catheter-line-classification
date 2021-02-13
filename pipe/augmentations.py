import albumentations
import albumentations.augmentations.transforms as A
from albumentations.core.composition import OneOf
from albumentations.pytorch import transforms


def augmentations_factory(hparams: str):
    mean = 0.485
    std = 0.2295

    # TODO: more elegant way of doing this
    input_image_sz = int(hparams.train_data.name.split("_")[1])

    if hparams.aug == "baseline":
        train_augmentation = albumentations.Compose(
            [
                albumentations.CLAHE(p=1),
                albumentations.HorizontalFlip(p=0.25),
                albumentations.OneOf(
                    [
                        albumentations.RandomBrightnessContrast(
                            brightness_limit=0.1, contrast_limit=0.1
                        ),
                    ],
                    p=0.5,
                ),
                albumentations.OneOf(
                    [
                        albumentations.OpticalDistortion(),
                        albumentations.ElasticTransform(),
                    ],
                    p=0.25,
                ),
                albumentations.Cutout(
                    p=0.25,
                    num_holes=8,
                    max_h_size=int(hparams.sz * 0.1),
                    max_w_size=int(hparams.sz * 0.1),
                ),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8
                ),
                albumentations.RandomCrop(
                    int(input_image_sz * 0.9), int(input_image_sz * 0.9)
                ),
                albumentations.Resize(height=hparams.sz, width=hparams.sz),
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                transforms.ToTensorV2(),
            ]
        )
        valid_augmentation = albumentations.Compose(
            [
                albumentations.CLAHE(p=1),
                albumentations.CenterCrop(
                    int(input_image_sz * 0.9), int(input_image_sz * 0.9)
                ),
                albumentations.Resize(height=hparams.sz, width=hparams.sz),
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                transforms.ToTensorV2(),
            ]
        )
        test_augmentation = albumentations.Compose(
            [
                albumentations.CLAHE(p=1),
                albumentations.CenterCrop(int(256 * 0.9), int(256 * 0.9)),
                albumentations.Resize(height=hparams.sz, width=hparams.sz),
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                transforms.ToTensorV2(),
            ]
        )
        return train_augmentation, valid_augmentation, test_augmentation
    if hparams.aug == "no_clahe":
        train_augmentation = albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=0.25),
                albumentations.OneOf(
                    [
                        albumentations.RandomBrightnessContrast(
                            brightness_limit=0.1, contrast_limit=0.1
                        ),
                    ],
                    p=0.5,
                ),
                albumentations.OneOf(
                    [
                        albumentations.OpticalDistortion(),
                        albumentations.ElasticTransform(),
                    ],
                    p=0.25,
                ),
                albumentations.Cutout(
                    p=0.25,
                    num_holes=8,
                    max_h_size=int(hparams.sz * 0.1),
                    max_w_size=int(hparams.sz * 0.1),
                ),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8
                ),
                albumentations.RandomCrop(
                    int(input_image_sz * 0.9), int(input_image_sz * 0.9)
                ),
                albumentations.Resize(height=hparams.sz, width=hparams.sz),
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                transforms.ToTensorV2(),
            ]
        )
        valid_augmentation = albumentations.Compose(
            [
                albumentations.CenterCrop(
                    int(input_image_sz * 0.9), int(input_image_sz * 0.9)
                ),
                albumentations.Resize(height=hparams.sz, width=hparams.sz),
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                transforms.ToTensorV2(),
            ]
        )
        test_augmentation = albumentations.Compose(
            [
                albumentations.CenterCrop(int(256 * 0.9), int(256 * 0.9)),
                albumentations.Resize(height=hparams.sz, width=hparams.sz),
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                transforms.ToTensorV2(),
            ]
        )
        return train_augmentation, valid_augmentation, test_augmentation
    if hparams.aug == "50_pct_clahe":
        train_augmentation = albumentations.Compose(
            [
                albumentations.CLAHE(p=0.5),
                albumentations.HorizontalFlip(p=0.25),
                albumentations.OneOf(
                    [
                        albumentations.RandomBrightnessContrast(
                            brightness_limit=0.1, contrast_limit=0.1
                        ),
                    ],
                    p=0.5,
                ),
                albumentations.OneOf(
                    [
                        albumentations.OpticalDistortion(),
                        albumentations.ElasticTransform(),
                    ],
                    p=0.25,
                ),
                albumentations.Cutout(
                    p=0.25,
                    num_holes=8,
                    max_h_size=int(hparams.sz * 0.1),
                    max_w_size=int(hparams.sz * 0.1),
                ),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8
                ),
                albumentations.RandomCrop(
                    int(input_image_sz * 0.9), int(input_image_sz * 0.9)
                ),
                albumentations.Resize(height=hparams.sz, width=hparams.sz),
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                transforms.ToTensorV2(),
            ]
        )
        valid_augmentation = albumentations.Compose(
            [
                albumentations.CLAHE(p=0.5),
                albumentations.CenterCrop(
                    int(input_image_sz * 0.9), int(input_image_sz * 0.9)
                ),
                albumentations.Resize(height=hparams.sz, width=hparams.sz),
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                transforms.ToTensorV2(),
            ]
        )
        test_augmentation = albumentations.Compose(
            [
                albumentations.CLAHE(p=0.5),
                albumentations.CenterCrop(int(256 * 0.9), int(256 * 0.9)),
                albumentations.Resize(height=hparams.sz, width=hparams.sz),
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                transforms.ToTensorV2(),
            ]
        )
        return train_augmentation, valid_augmentation, test_augmentation
    raise ValueError("Augmentation config not recognized")
