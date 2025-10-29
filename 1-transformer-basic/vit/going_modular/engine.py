import torch

import torch.utils
import torch.utils.tensorboard
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    trains a torch model for a single epoch

    Args:
        model (torch.nn.Module): _description_
        dataloader (torch.utils.data.DataLoader): _description_
        loss_fn (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        device (torch.device): _description_

    Returns:
        Tuple[float,float]: (train_loss, train_accuracy). for example:
        (0.112, 0.8743)
    """
    model.train()
    train_loss, train_acc = 0, 0
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 1 forward pass
        y_pred = model(X)
        # 2 loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # 3 optim
        optimizer.zero_grad()
        # 4 back
        loss.backward()
        # 5 optimizer step
        optimizer.step()

        # accumulate acc
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # avarge loss and acc
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    test a torch model for a single epoch

    Args:
        model (torch.nn.Module): _description_
        dataloader (torch.utils.data.DataLoader): _description_
        loss_fn (torch.nn.Module): _description_
        device (torch.device): _description_

    Returns:
        Tuple[float,float]: (test_loss, test_accuracy). for example:
        (0.112, 0.8743)
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # 1 forward pass
            y_pred = model(X)
            # 2 loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            # accumulate acc
            y_pred_class = torch.argmax(y_pred, dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # avarge loss and acc
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: torch.utils.tensorboard.writer.SummaryWriter = None,
) -> Dict[str, List]:
    """
    trans and tests a torch model

    Args:
        model (torch.nn.Module): _description_
        train_dataloader (torch.utils.data.DataLoader): _description_
        test_dataloader (torch.utils.data.DataLoader): _description_
        optimizer (torch.optim.Optimizer): _description_
        loss_fn (torch.nn.Module): _description_
        epochs (int): _description_
        device (torch.device): _description_
        writer

    Returns:
        Dict[str,List]: a dictionary of training and testing loss as well as training and testing acc metrics. each metric as a value in a list for each epoch
        For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
    """
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        print(
            f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                global_step=epoch,
            )
            writer.close()

    return results
