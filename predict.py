import json

import model_utils
import input_utils
import data_utils


args = input_utils.get_predict_args()

top_k = args.top_k
category_names = args.category_names
gpu = False if args.gpu is None else True
checkpoint = args.checkpoint
category_names = args.category_names

model = model_utils.get_loaded_model(checkpoint)
probs, predict_classes = model_utils.get_prediction(
    model, data_utils.process_image(args.image_path), top_k
)

with open(category_names, "r") as f:
    cat_to_name = json.load(f)

    for prob, predict_classe in zip(probs, predict_classes):
        print(prob)
        print(predict_classes)
        print(cat_to_name[predict_classe])
