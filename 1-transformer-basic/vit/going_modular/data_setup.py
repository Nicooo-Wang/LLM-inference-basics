import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_OF_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_OF_WORKERS,
):
    """_summary_

    Args:
        train_dir (str): _description_
        test_dir (str): _description_
        train_transform (transforms.Compose): _description_
        test_transform (transforms.Compose): _description_
        batch_size (int): _description_
        num_workers (int, optional): _description_. Defaults to NUM_OF_WORKERS.
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
        Example usage:
            train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
                                                                                test_dir=path/to/test_dir,
                                                                                transform=some_trans,
                                                                                batch_size=32,
                                                                                num_workers=4)
    """
    # datasets
    train_data = datasets.ImageFolder(train_dir,transform=train_transform)
    test_data = datasets.ImageFolder(test_dir,transform=test_transform)

    # class_names
    res_class_names = train_data.classes

    # dataloaders
    res_train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    res_test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return res_train_dataloader,res_test_dataloader,res_class_names
