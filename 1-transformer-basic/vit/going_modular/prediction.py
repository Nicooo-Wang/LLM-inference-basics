from typing import List, Tuple
import matplotlib.pyplot as plt
from PIL import Image
from  tqdm import tqdm
import torch
from going_modular import data_setup, model_builder, engine, utils
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pred_and_plt_confmat(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
):
    """
    predict on test set and plot confmat

    Args:
        model (torch.nn.Module): _description_
        test_transform (_type_): _description_
    """
    y_pred, y_target = pred_on_test_set(
        model=model, dataloader=test_dataloader, device=device
    )

    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass").to(
        device
    )
    confmat_tensor = confmat(preds=y_pred, target=y_target)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.to("cpu").numpy(), class_names=class_names
    )
    plt.show()


def plot_loss_curves(results: dict):
    """
    plot training curves of a results dictionary

    Args:
        results (dict): dict containing list of values
        {
            "train_loss": [...],
            "test_loss": [...],
            "train_acc": [...],
            "test_acc": [...]
        }
    """
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    train_acc = results["train_acc"]
    test_acc = results["test_acc"]

    epochs = range(len(train_loss))
    plt.figure(figsize=(15, 7))

    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # plot acc
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, test_acc, label="test_acc")
    plt.title("acc")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()

def pred_on_test_set(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    test a torch model on test set

    Args:
        model (torch.nn.Module): _description_
        dataloader (torch.utils.data.DataLoader): _description_
        loss_fn (torch.nn.Module): _description_
        device (torch.device): _description_

    Returns:
        Tuple[torch.Tensor,torch.Tensor]: (test_pred, test_targets).
    """
    model.eval()
    y_pred_lst, y_lst = [], []
    with torch.inference_mode():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred_lst.append(y_pred)
            y_lst.append(y)

    y_pred, y = torch.concat(y_pred_lst, dim=0), torch.concat(y_lst, dim=0)
    y_pred, y = y_pred.to(device), y.to(device)
    return y_pred, y

def pred_and_plt_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    transform: torchvision.transforms,
    device: torch.device = DEVICE,
):
    """
    _summary_

    Args:
        model (torch.nn.Module): _description_
        image_path (str): _description_
        class_names (List[str]): _description_
        image_size (Tuple[int, int]): _description_
        transform (torchvision.transforms, optional): _description_. Defaults to None.
        device (torch.device, optional): _description_. Defaults to DEVICE.
    """
    # 2 open image
    img_pil = Image.open(image_path)
    # 3. pred
    model.to(device)
    model.eval()
    with torch.inference_mode():
        X = transform(img_pil).unsqueeze(0).to(device)
        y_pred = model(X)
    y_pred_probs = torch.softmax(y_pred, dim=1)
    y_pred_label = torch.argmax(y_pred_probs, dim=1)
    plt.figure()
    plt.imshow(img_pil)
    plt.title(f"pred: {class_names[y_pred_label]} | probility: {y_pred_probs.max()}")
    plt.axis(False)
    plt.show()
