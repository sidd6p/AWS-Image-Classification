def get_model_args():
    data = get_input_args()

    # Hyperparmeters
    models_args = dict()
    models_args["batch_size"] = 64
    models_args["epochs"] = data.epochs
    models_args["learning_rate"] = data.learning_rate
    models_args["criterion"] = nn.NLLLoss()

    # Other constants
    models_args["input_size"] = int(data.hidden_units)
    models_args["output_size"] = 102
    models_args["hidden_layers"] = [512]
    models_args["drop_rate"] = 0.5
    models_args["arch"] = data.arch
    models_args["device"] = torch.device(
        "cuda:0" if torch.cuda.is_available() and data.gpu else "cpu"
    )

    # Data
    models_args["save_dir"] = data.save_dir
    models_args["data_dir"] = data.data_directory

    # Setting data paths
    models_args["train_dir"] = models_args["data_dir"] + "/train"
    models_args["valid_dir"] = models_args["data_dir"] + "/valid"
    models_args["test_dir"] = models_args["data_dir"] + "/test"

    return models_args


def get_transformer():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # transforms for the training, validation, and testing sets
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

    return data_transforms


def get_image_datasets(models_args):
    image_datasets = {
        "train": datasets.ImageFolder(
            models_args["train_dir"], transform=data_transforms["train"]
        ),
        "valid": datasets.ImageFolder(
            models_args["valid_dir"], transform=data_transforms["valid"]
        ),
        "test": datasets.ImageFolder(
            models_args["test_dir"], transform=data_transforms["test"]
        ),
    }
    return image_datasets


def get_dataloaders(image_datasets):
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"], models_args["batch_size"], shuffle=True
        ),
        "val_lovalidader": torch.utils.data.DataLoader(
            image_datasets["valid"], models_args["batch_size"], shuffle=True
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"], batch_size=32, shuffle=True
        ),
    }
    return dataloaders


def get_model(models_args):
    # Select the model.
    model_data = getattr(models, models_args["arch"])
    user_model = model_data(pretrained=True)

    for param in user_model.parameters():
        param.requires_grad = False

    return user_model


def get_optimizer(model, models_args):
    return optim.Adam(model.fc.parameters(), lr=models_args["learn_rate"])


import argparse
import torch

import torch.nn.functional as F


from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", action="store", help="Set path to folder")
    parser.add_argument(
        "--save_dir", type=str, default="/", help="Set directory to save checkpoints"
    )
    parser.add_argument(
        "--arch", type=str, default="vgg", help="Set model architecture"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Set learning rate hyperparameter",
    )
    parser.add_argument(
        "--hidden_units", type=int, default=512, help="Set hidden units hyperparameter"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Set epochs hyperparameter"
    )
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="Set GPU")

    return parser.parse_args()


# Defining Classifier
# class MyNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()


#         self.hidden_layers = nn.ModuleList([nn.Linear( models_args["input_size"], models_args['hidden_layers'][0])])
#         layer_sizes = zip(models_args['hidden_layers'][:-1], models_args['hidden_layers'][1:])
#         self.hidden_layers.extend([nn.Linear(h1,h2) for h1, h2 in layer_sizes])
#         self.output = nn.Linear(models_args['hidden_layers'][-1], models_args["output_size"])
#         self.dropout = nn.Dropout(p = models_args['drop_rate'])

#         # self.layer1 = nn.Linear(
#         #     models_args["input_size"], models_args["input_size"] // 2
#         # )
#         # self.layer2 = nn.Linear(
#         #     models_args["input_size"] // 2, models_args["input_size"] // 4
#         # )
#         # self.layer3 = nn.Linear(
#         #     models_args["input_size"] // 4, models_args["output_size"]
#         # )

#         # self.dropout = nn.Dropout(models_args["drop_rate"])

#         # self.output = nn.LogSoftmax(dim=1)

#     def forward(self, xb):
#         for layer in self.hidden_layers:
#             xb = F.relu(layer(xb))
#             xb = self.dropout(xb)
#         out = self.output(xb)

#         return F.log_softmax(out, dim= 1)
#         # out = self.linear1(x)
#         # out = F.ReLU(out)
#         # out = self.dropout(out)

#         # out = self.linear2(out)
#         # out = F.ReLU(out)

#         # out = self.linear3(out)

#         # out = self.output(out)

#         # return out


# Defining Classifier
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(models_args["input_size"], models_args["hidden_layers"][0])]
        )
        layer_sizes = zip(
            models_args["hidden_layers"][:-1], models_args["hidden_layers"][1:]
        )
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(
            models_args["hidden_layers"][-1], models_args["output_size"]
        )
        self.dropout = nn.Dropout(p=models_args["drop_rate"])

    def forward(self, xb):
        for layer in self.hidden_layers:
            xb = F.relu(layer(xb))
            xb = self.dropout(xb)
        out = self.output(xb)
        return F.log_softmax(out, dim=1)


def validation_model_train(model, criterion, data, models_args):
    loss, accuracy = 0, 0
    model.to(models_args["device"])

    for images, labels in data:
        images, labels = images.to(models_args["device"]), labels.to(
            models_args["device"]
        )
        output = model(images)  # Generate predictions
        loss += criterion(output, labels).item()  # Calculate loss
        ps = torch.exp(output)
        equity = labels == ps.max(dim=1)[1]
        accuracy += equity.type(torch.FloatTensor).mean()

    return loss, accuracy


# train the network
def trainthemodel(model, optimizer, train_dataloaders, valid_dataloaders, models_args):
    steps = 0

    model.to(models_args["device"])

    for epoch in range(models_args["epochs"]):
        model.train()

        for images, labels in train_dataloaders:
            steps += 1
            images, labels = images.to(models_args["device"]), labels.to(
                models_args["device"]
            )
            optimizer.zero_grad()
            output = model.forward(images)
            print(criterion)
            Loss = criterion(output, labels)

            Loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():  # so that gradient do not get re-calculate
            loss, accuracy, totaldatalen = validation_model_train(
                model, criterion, valid_dataloaders, models_args
            )

        print(
            "Epoch: {}/{}\nLoss: {:.4f}\nTest Accuracy: {:.3f}%\n\n".format(
                epoch + 1,
                models_args["epochs"],
                loss / len(valid_dataloaders),
                accuracy / len(valid_dataloaders) * 100,
            )
        )

        model.train()


# def train_model(model, optimizer, train_dataloaders, valid_dataloaders, models_args):
#     steps = 0
#     model.to(models_args["device"])
#     for epoch in range(models_args["epochs"]):
#         model.train()
#         for images, labels in train_dataloaders:
#             steps += 1
#             images, labels = images.to(models_args["device"]), labels.to(
#                 models_args["device"]
#             )
#             optimizer.zero_grad()
#             output = model.forward(images)
#             print(output)
#             print(labels)
#             Loss = criterion(output, labels)

#             Loss.backward()
#             optimizer.step()

#         model.eval()
#         with torch.no_grad():  # so that gradient do not get re-calculate
#             loss, accuracy = validation_model_train(
#                 model, criterion, valid_dataloaders, models_args
#             )
#         print(
#             "Epoch: {}/{}\nLoss: {:.4f}\nTest Accuracy: {:.3f}%\n\n".format(
#                 epoch + 1,
#                 models_args["epochs"],
#                 loss / len(valid_dataloaders),
#                 accuracy / len(valid_dataloaders) * 100,
#             )
#         )

#         model.train()


def validation_on_test(model, test_dataloaders, models_args):
    accuracy, data_size = 0, 0
    model.to(models_args["device"])

    model.eval()
    for images, labels in test_dataloaders:
        images, labels = images.to(models_args["device"]), labels.to(
            models_args["device"]
        )
        output = model.forward(images)
        ps = torch.exp(output)
        equity = labels == ps.max(dim=1)[1]
        data_size += labels.size(0)
        accuracy += equity.type(torch.FloatTensor).sum().item()

    return accuracy, data_size


def get_accuracy(model, criterion, test_loader):
    with torch.no_grad():
        accuracy, total = validation_on_test(model, criterion, test_loader)
    print("Accuracy: {}%\n".format(100 * accuracy / total))


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    # models_args = get_model_args()
    data = get_input_args()

    # Hyperparmeters
    models_args = dict()
    models_args["batch_size"] = 64
    models_args["epochs"] = data.epochs
    models_args["learning_rate"] = data.learning_rate
    criterion = nn.NLLLoss()

    # Other constants
    models_args["input_size"] = int(data.hidden_units)
    models_args["output_size"] = 102
    models_args["hidden_layers"] = [512]
    models_args["drop_rate"] = 0.5
    models_args["arch"] = data.arch
    models_args["device"] = torch.device(
        "cuda:0" if torch.cuda.is_available() and data.gpu else "cpu"
    )

    # Data
    models_args["save_dir"] = data.save_dir
    models_args["data_dir"] = data.data_directory

    # Setting data paths
    models_args["train_dir"] = models_args["data_dir"] + "/train"
    models_args["valid_dir"] = models_args["data_dir"] + "/valid"
    models_args["test_dir"] = models_args["data_dir"] + "/test"

    # data_transforms = get_transformer()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # transforms for the training, validation, and testing sets
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

    # image_datasets = get_image_datasets(models_args)
    image_datasets = {
        "train": datasets.ImageFolder(
            models_args["train_dir"], transform=data_transforms["train"]
        ),
        "valid": datasets.ImageFolder(
            models_args["valid_dir"], transform=data_transforms["valid"]
        ),
        "test": datasets.ImageFolder(
            models_args["test_dir"], transform=data_transforms["test"]
        ),
    }

    # dataloaders = get_dataloaders(image_datasets)
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"], models_args["batch_size"], shuffle=True
        ),
        "val": torch.utils.data.DataLoader(
            image_datasets["valid"], models_args["batch_size"], shuffle=True
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"], batch_size=32, shuffle=True
        ),
    }

    # user_model = get_model(models_args)
    # Select the model.
    model_types = {
        "resnet": models.resnet18(pretrained=True),
        "alexnet": models.alexnet(pretrained=True),
        "vgg": models.vgg16(pretrained=True),
    }
    user_model = model_types[models_args["arch"]]

    for param in user_model.parameters():
        param.requires_grad = False

    my_classifier = MyModel()

    user_model.fc = my_classifier

    # optimizer = get_optimizer(user_model, models_args)
    optimizer = optim.Adam(user_model.fc.parameters(), lr=models_args["learning_rate"])

    trainthemodel(
        user_model,
        optimizer,
        dataloaders["train"],
        dataloaders["val"],
        models_args,
    )

    get_accuracy(user_model, criterion, dataloaders["test_loader"])


import argparse
import torch

import torch.nn.functional as F


from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", action="store", help="Set path to folder")
    parser.add_argument(
        "--save_dir", type=str, default="/", help="Set directory to save checkpoints"
    )
    parser.add_argument(
        "--arch", type=str, default="vgg", help="Set model architecture"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Set learning rate hyperparameter",
    )
    parser.add_argument(
        "--hidden_units", type=int, default=512, help="Set hidden units hyperparameter"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Set epochs hyperparameter"
    )
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="Set GPU")

    return parser.parse_args()


models_args = dict()


# Defining Classifier
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(models_args["input_size"], models_args["hidden_layers"][0])]
        )
        layer_sizes = zip(
            models_args["hidden_layers"][:-1], models_args["hidden_layers"][1:]
        )
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(
            models_args["hidden_layers"][-1], models_args["output_size"]
        )
        self.dropout = nn.Dropout(p=models_args["drop_rate"])

    def forward(self, xb):
        for layer in self.hidden_layers:
            xb = F.relu(layer(xb))
            xb = self.dropout(xb)
        out = self.output(xb)

        return F.log_softmax(out, dim=1)


def validation(model, criterion, data):
    loss, accuracy, data_len = 0, 0, 0
    model.to(models_args["device"])

    for images, labels in data:
        images, labels = images.to(models_args["device"]), labels.to(
            models_args["device"]
        )
        output = model(images)  # Generate predictions
        loss += criterion(output, labels).item()  # Calculate loss
        ps = torch.exp(output)
        equity = labels == ps.max(dim=1)[1]
        data_len += labels.size(0)
        accuracy += equity.type(torch.FloatTensor).mean()

    return loss, accuracy, data_len


# train the network
def trainthemodel(model, optimizer, train_dataloaders, valid_dataloaders):
    steps = 0

    model.to(models_args["device"])

    for epoch in range(models_args["epochs"]):
        model.train()

        for images, labels in train_dataloaders:
            steps += 1
            images, labels = images.to(models_args["device"]), labels.to(
                models_args["device"]
            )
            optimizer.zero_grad()
            output = model.forward(images)
            Loss = criterion(output, labels)

            Loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():  # so that gradient do not get re-calculate
            loss, accuracy, totaldatalen = validation(
                model, criterion, valid_dataloaders
            )

        print(
            "Epoch: {}/{}\nLoss: {:.4f}\nTest Accuracy: {:.3f}%\n\n".format(
                epoch + 1,
                models_args["epochs"],
                loss / len(valid_dataloaders),
                accuracy / len(valid_dataloaders) * 100,
            )
        )

        model.train()


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    data = get_input_args()

    models_args["batch_size"] = 64
    models_args["epochs"] = data.epochs
    models_args["learning_rate"] = data.learning_rate
    criterion = nn.NLLLoss()

    models_args["input_size"] = 2048
    models_args["output_size"] = 102
    models_args["hidden_layers"] = [512]
    models_args["drop_rate"] = 0.5
    models_args["arch"] = data.arch
    models_args["device"] = torch.device(
        "cuda:0" if torch.cuda.is_available() and data.gpu else "cpu"
    )

    models_args["save_dir"] = data.save_dir
    models_args["data_dir"] = data.data_directory

    models_args["train_dir"] = models_args["data_dir"] + "/train"
    models_args["valid_dir"] = models_args["data_dir"] + "/valid"
    models_args["test_dir"] = models_args["data_dir"] + "/test"

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
        "train": datasets.ImageFolder(
            models_args["train_dir"], transform=data_transforms["train"]
        ),
        "valid": datasets.ImageFolder(
            models_args["valid_dir"], transform=data_transforms["valid"]
        ),
        "test": datasets.ImageFolder(
            models_args["test_dir"], transform=data_transforms["test"]
        ),
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"], models_args["batch_size"], shuffle=True
        ),
        "val": torch.utils.data.DataLoader(
            image_datasets["valid"], models_args["batch_size"], shuffle=True
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"], batch_size=32, shuffle=True
        ),
    }

    if models_args["arch"] == "resnet":
        user_model = models.resnet50(pretrained=True)
    elif models_args["arch"] == "alexnet":
        user_model = models.alexnet(pretrained=True)
    else:
        user_model = models.vgg16(pretrained=True)

    for param in user_model.parameters():
        param.requires_grad = False

    my_classifier = MyModel()
    my_classifier
    user_model.fc = my_classifier
    optimizer = optim.Adam(user_model.fc.parameters(), lr=models_args["learning_rate"])

    trainthemodel(user_model, optimizer, dataloaders["train"], dataloaders["val"])
