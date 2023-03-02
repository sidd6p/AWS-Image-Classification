import argparse
from data_utils import load_data
import model_utils
import input_utils

parser = argparse.ArgumentParser(
    description="Training a neural network on a given dataset"
)
parser.add_argument(
    "data_directory",
    help="Path to dataset on which the neural network should be trained on",
)
parser.add_argument(
    "--save_dir", help="Path to directory where the checkpoint should be saved"
)
parser.add_argument("--arch", help="Network architecture (default 'vgg16')")
parser.add_argument("--learning_rate", help="Learning rate")
parser.add_argument("--hidden_units", help="Number of hidden units")
parser.add_argument("--epochs", help="Number of epochs")
parser.add_argument("--gpu", help="Use GPU for training", action="store_true")


args = input_utils.get_input_args()

data_directory = args.data_directory
save_dir = args.save_dir
network_architecture = args.arch
learning_rate = args.learning_rate
hidden_units = int(args.hidden_units)
epochs = int(args.epochs)
gpu = False if args.gpu is None else True


train_data, trainloader, validloader, testloader = load_data(data_directory)


model = model_utils.build_network(network_architecture, hidden_units)
model.class_to_idx = train_data.class_to_idx

model, criterion = model_utils.train_network(
    model, epochs, learning_rate, trainloader, validloader, gpu
)
model_utils.evaluate_model(model, testloader, criterion, gpu)
model_utils.save_model(
    model, network_architecture, hidden_units, epochs, learning_rate, save_dir
)
