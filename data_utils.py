import torch
from torchvision import datasets, transforms, models
from PIL import Image


def get_data(path):
    train_dir = path + "/train"
    valid_dir = path + "/valid"
    test_dir = path + "/test"

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    data_transforms = {}

    data_transforms["train"] = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.ToTensor(), normalize]
    )

    data_transforms["valid"] = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    data_transforms["test"] = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"], batch_size=64, shuffle=True
        ),
        "val": torch.utils.data.DataLoader(
            image_datasets["valid"], batch_size=64, shuffle=True
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"], batch_size=64, shuffle=True
        ),
    }

    return image_datasets, dataloaders


def process_image(image):
    image = Image.open(image)

    image_transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return image_transform(image)
