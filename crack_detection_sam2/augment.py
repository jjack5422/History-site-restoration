import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(image_size=512, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.6, 1.0),
                            ratio=(0.75, 1.33), interpolation=1, mask_interpolation=0, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-15, 15),
                 interpolation=1, mask_interpolation=0, p=0.5),
        A.ElasticTransform(alpha=40, sigma=6, interpolation=1, mask_interpolation=0, p=0.3),
        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def val_transforms(image_size=512, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, fill=0, fill_mask=0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
