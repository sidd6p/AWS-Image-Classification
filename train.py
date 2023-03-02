import model_utils
import input_utils

from data_utils import get_data


args = input_utils.get_input_args()

data_directory = args.data_directory
save_dir = args.save_dir
network_architecture = args.arch
learning_rate = args.learning_rate
hidden_units = int(args.hidden_units)
epochs = int(args.epochs)
gpu = False if args.gpu is None else True


image_datasets, dataloaders = get_data(data_directory)


model = model_utils.build_network(network_architecture, hidden_units)
model.class_to_idx = image_datasets["train"].class_to_idx

model, criterion = model_utils.train_network(
    model, epochs, learning_rate, dataloaders["train"], dataloaders["val"], gpu
)
model_utils.evaluate_model(model, dataloaders["test"], criterion, gpu)
model_utils.save_model(
    model, network_architecture, hidden_units, epochs, learning_rate, save_dir
)
