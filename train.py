import model_utils
import input_utils
import data_utils


args = input_utils.get_train_args()

data_directory = args.data_directory
save_dir = args.save_dir
architecture = args.arch
learning_rate = args.learning_rate
hidden_units = int(args.hidden_units)
epochs = int(args.epochs)
gpu = False if args.gpu is None else True
drop_rate = 0.2
output_size = 3

image_datasets, dataloaders = data_utils.get_data(data_directory)


model = model_utils.get_network(architecture, hidden_units, drop_rate, output_size)
model.class_to_idx = image_datasets["train"].class_to_idx

model = model_utils.train_network(
    model, epochs, learning_rate, dataloaders["train"], dataloaders["val"], gpu
)
model_utils.get_test_validation(model, dataloaders["test"], gpu)
model_utils.save_model(model, architecture, save_dir, image_datasets["train"])
