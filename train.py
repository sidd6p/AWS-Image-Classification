import argparse


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


def main():
    data = get_input_args()
    print(data)


if __name__ == "__main__":
    main()
