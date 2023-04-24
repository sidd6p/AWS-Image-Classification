# Image Classifier - Part 2 - Command Line App

## Part 2 - Building the command line application

- train.py, will train a new network on a dataset and save the model as a checkpoint.

- predict.py, uses a trained network to predict the class for an input image.

### Train
Train a new network on a data set with train.py

- Basic usage: python train.py data_directory
- Options: 
  - __Set directory to save checkpoints__: ```py python train.py data_dir --save_dir save_directory ```
  - __Coose architecture__: ```py python train.py data_dir --arch "vgg13" ```
    - These 4 architecture options are available
      1. vgg16
      2. vgg13
      3. alexnet
      4. densenet
  - __Set hyperparameters__: ```py python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 ```
  - __Use GPU for training__: ```py python train.py data_dir --gpu ```
  
### Predict
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

- Basic usage: python predict.py /path/to/image checkpoint
- Options: 
  - __Return top  K most likely classes__: ```py python predict.py input checkpoint --top_k 3 ```
  - __Use a mapping of categories to real names__: ```py python predict.py input checkpoint --category_names cat_to_name.json ```
  - __Use GPU for inference__: ```py python predict.py input checkpoint --gpu ```
