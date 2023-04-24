import utils as utils

def main():
    args = utils.input_utils.get_train_args()

    data_directory = args.data_directory
    save_dir = args.save_dir
    architecture = args.arch
    learning_rate = args.learning_rate
    hidden_units = int(args.hidden_units)
    epochs = int(args.epochs)
    gpu = False if args.gpu is None else True
    drop_rate = 0.2
    output_size = int(args.classes)
    image_datasets, dataloaders = utils.data_utils.get_data(data_directory)

    model = utils.model_utils.get_network(architecture, hidden_units, drop_rate, output_size)
    model.class_to_idx = image_datasets["train"].class_to_idx

    model = utils.model_utils.train_network(
        model, epochs, learning_rate, dataloaders["train"], dataloaders["val"], gpu
    )
    utils.model_utils.get_test_validation(model, dataloaders["test"], gpu)
    utils.model_utils.save_model(model, architecture, save_dir, image_datasets["train"])


if __name__ == "__main__":
    main()
