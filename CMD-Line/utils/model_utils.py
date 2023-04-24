import torch
from . import network_utils

from torch import nn
from torch import optim
from torchvision import models

criterion = nn.NLLLoss()


def get_network(architecture, hidden_units, drop_rate=0.2, output_size=3):
    if architecture == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif architecture == "vgg13":
        model = models.vgg13(pretrained=True)
        input_size = 25088
    elif architecture == "alexnet":
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif architecture == "densenet":
        model = models.densenet121(pretrained=True)
        input_size = 1024

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = network_utils.MyNetwork(
        input_size, hidden_units, drop_rate, output_size
    )

    return model


def get_train_validation(model, data, device):
    loss, accuracy = 0, 0
    model.to(device)
    validloader = data

    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)  # Generate predictions
        loss += criterion(output, labels).item()  # Calculate loss
        ps = torch.exp(output)
        equity = labels == ps.max(dim=1)[1]
        accuracy += equity.type(torch.FloatTensor).mean()

    return loss, accuracy


def get_test_validation(model, test_dataloaders, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    accuracy, data_size = 0, 0

    model.to(device)

    model.eval()
    for images, labels in test_dataloaders:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        ps = torch.exp(output)
        equity = labels == ps.max(dim=1)[1]
        data_size += labels.size(0)
        accuracy += equity.type(torch.FloatTensor).sum().item()

    print("Accuracy: {}%\n".format(100 * accuracy / data_size))


def train_network(model, epochs, learning_rate, trainloader, validloader, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    steps = 0

    for epoch in range(epochs):
        training_loss = 0
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            Loss = criterion(output, labels)

            Loss.backward()
            optimizer.step()

            training_loss += Loss.item()

        model.eval()
        with torch.no_grad():  # so that gradient do not get re-calculate
            loss, accuracy = get_train_validation(model, validloader, device)

        print(
            "Epoch: {}/{}\nTraining Loss: {:.4f}\nValidation Loss: {:.4f}\nvalidation  Accuracy: {:.3f}%\n\n".format(
                epoch + 1,
                epochs,
                training_loss,
                loss / len(validloader),
                accuracy / len(validloader) * 100,
            )
        )

        model.train()

    return model


def save_model(model, architecture, save_dir, data):
    checkpoint = {
        "architecture": architecture,
        "model_state_dict": model.state_dict(),
        "class_to_idx": data.class_to_idx,
        "model": model,
        "classifier": model.classifier,
    }

    checkpoint_path = save_dir + "checkpoint.pth"

    torch.save(checkpoint, checkpoint_path)


def get_loaded_model(checkpoint):
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    checkpoint = torch.load(checkpoint, map_location=map_location)

    model = checkpoint["model"]
    model.class_to_idx = checkpoint["class_to_idx"]
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["model_state_dict"])

    for param in model.parameters():
        param.requires_grad = False

    return model


def get_prediction(model, image, top_k, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model, image = model.to(device), image.to(device)
    model.eval()

    with torch.no_grad():
        results = torch.exp(model.forward(image.unsqueeze(0).float()))
        result_top_k = results.topk(top_k)
        probs, classes = (
            result_top_k[0].data.cpu().numpy()[0],
            result_top_k[1].data.cpu().numpy()[0],
        )

        idx_to_class = {key: value for value, key in model.class_to_idx.items()}
        classes = [idx_to_class[classes[i]] for i in range(classes.size)]

    return probs, classes
