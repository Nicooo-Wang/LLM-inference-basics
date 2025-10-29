import torch
import matplotlib.pyplot as plt
import torch.utils
from going_modular import engine
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torchvision
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    save torch model to a target directory

    Args:
        model (torch.nn.Module): _description_
        target_dir (str): _description_
        model_name (str): _description_

    Example Usage:
        save_model(model_0,target_dir="models",model_name="tinyVGG_model.pth)
    """

    assert model_name.endswith(".pth") or model_name.endswith(".pt")

    dir_path = Path(target_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    model_path = dir_path / model_name

    print(f"Saving model to: {model_path}")
    torch.save(obj=model.state_dict(), f=model_path)


def create_summary_writer(
    experiment_name: str, model_name: str, extra: str = None
) -> torch.utils.tensorboard.writer.SummaryWriter:
    """
    creates a torch.utils.tensorboard.writer.SummaryWriter()

    Args:
        experiment_name (str): _description_
        model_name (str): _description_
        extra (str, optional): _description_. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter: _description_
    """
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y-%m-%d")

    log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    if extra:
        log_dir = os.path.join(log_dir, extra)
    print(f"SummaryWriter Created, saving to : {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
