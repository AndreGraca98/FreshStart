import copy
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

DEVICE = "cuda"
LR = 1e3
BATCHSIZE = 128
EPOCHS = 3

torch.backends.cudnn.enabled = False
torch.manual_seed(0)


def get_loaders(batchsize):
    train_loader = torch.utils.data.DataLoader(
        MNIST(
            "files/",
            train=True,
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batchsize,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        MNIST(
            "files/",
            train=False,
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batchsize,
        shuffle=True,
    )

    dataloaders = {"train": train_loader, "val": test_loader}
    dataset_sizes = {
        "train": len(train_loader.dataset),
        "val": len(test_loader.dataset),
    }

    return dataloaders, dataset_sizes


def build_net(lr, device):
    model = resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )  # Change conv1 to accept images with 1 channel
    num_features = model.fc.in_features  # extract fc layers features
    model.fc = nn.Linear(num_features, 10)  # (num_of_class == 10)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    return model, criterion, optimizer, scheduler


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    device,
    num_epochs=25,
):
    """From: https://www.kaggle.com/code/jokyeongmin/mnist-resnet18-in-pytorch/notebook"""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.upper()} "):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


if __name__ == "__main__":
    dataloaders, dataset_sizes = get_loaders(BATCHSIZE)
    model, criterion, optimizer, scheduler = build_net(LR, DEVICE)

    train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        DEVICE,
        EPOCHS,
    )

# ENDFILE
