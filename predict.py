import argparse


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_image", action="store", help="Give image path")
    parser.add_argument("checkpoint", action="store", help="Give checkpoint path")
    parser.add_argument(
        "--top_k", type=int, default=3, help="Return top K most likely classes"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="path to the JSON file",
    )
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="Set GPU")

    return parser.parse_args()


def main():
    data = get_input_args()
    print(data)


if __name__ == "__main__":
    main()
