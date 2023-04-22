import argparse


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", action="store", help="Set path to folder")
    parser.add_argument(
        "--save_dir", type=str, default="./", help="Set directory to save checkpoints"
    )
    parser.add_argument(
        "--arch", type=str, default="vgg16", help="Set model architecture"
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
        "--epochs", type=int, default=10, help="Set epochs hyperparameter"
    )
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="Set GPU")

    return parser.parse_args()


def get_predict_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", action="store", help="Set path to image")
    parser.add_argument("checkpoint", action="store", help="Set path to checkpoint")
    parser.add_argument(
        "--top_k", type=int, default=5, help="Set top k mostl likely classes"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="Use a mapping of categories to real names",
    )
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="Set GPU")

    return parser.parse_args()
