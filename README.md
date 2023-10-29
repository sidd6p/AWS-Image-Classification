# This is Image classification  project for the Amazon AI Programming with Python Nanodegree on Udacity
> This project use flower dataset for classification, but it can be use for any dataset as long as dataset is in correct format and structured into correct directory hierarchical. Look into "CMD-Line/flowers_dataset_small" for understanding the dateset hierarchical

![image](https://user-images.githubusercontent.com/91800813/234004940-2619b469-1e93-4b3b-893a-dfb54ac050a5.png)

[Certificate Link](https://drive.google.com/file/d/1NCuu7x-JIk7iJ8ZoSCSsGjbdZF3tAuyz/view?usp=share_link)

- There are two part in this project
  - Jupyter Notebook, where you run the whole project in jupyter notebook. Check out "Jupyter-Notebook" for more detail
  - Command Line, where you run whole project as a command line utility. Check out "CMD-Line" for more detail



# Image Classifier - Part 2 - Command Line App

## Directory stracture

    │   cat_to_name.json
    │   checkpoint.pth
    │   predict.py
    │   README.md
    │   requirements.txt
    │   train.py
    │
    ├───flowers_dataset_small
    │   ├───test
    │   │   ├───1
    │   │   │       image_06743.jpg
    │   │   │       ...............
    │   │   │
    │   │   ├───2
    │   │   │       image_05100.jpg
    │   │   │       ...............
    │   │   │
    │   │   └───3
    │   │           image_06634.jpg
    │   │           ...............
    │   │
    │   ├───train
    │   │   ├───1
    │   │   │       image_06734.jpg
    │   │   │       ...............
    │   │   │
    │   │   ├───2
    │   │   │       image_05087.jpg
    │   │   │       ...............
    │   │   │
    │   │   └───3
    │   │           image_06612.jpg
    │   │           ...............
    │   │
    │   └───valid
    │       ├───1
    │       │       image_06739.jpg
    │       │       ...............
    │       │
    │       ├───2
    │       │       image_05094.jpg
    │       │       ...............
    │       │
    │       └───3
    │               image_06621.jpg
    │               ...............
    │
    └───utils
            data_utils.py
            input_utils.py
            model_utils.py
            network_utils.py
            __init__.py
## Run Locally


- Clone the project

```bash
  git clone https://github.com/sidd6p/AWS-Image-Classification.git
```

- Go to the AWS-Image-Classification/CMD-Line directory
```bash
  cd AWS-Image-Classification/CMD-Line 

```
- Install requirements
```bash 
    pip install -r requirements.txt
```
- Train Model
```bash 
    python train.py data_dir 
```
- Make Prediction
```bash 
    python predict.py <path_to_image_file>/file_name <path_to_checkpoint_file>/checkpoint.pth
```

## Train
Train a new network on a data set with train.py

- Basic usage: python train.py data_directory
- Options: 
  - __Set directory to save checkpoints__: ``` python train.py data_dir --save_dir save_directory ```
  - __Choose architecture__: ```python train.py data_dir --arch "vgg13" ```
    - These 4 architecture options are available
      1. vgg16
      2. vgg13
      3. alexnet
      4. densenet
  - __Set hyperparameters__: ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 ```
  - __Use GPU for training__: ```python train.py data_dir --gpu ```
  
 ![image](https://user-images.githubusercontent.com/91800813/233997211-20a8e62b-a4e2-420c-bfd8-7d4b0a55a1d7.png)
 > This model is performing like this becaause I used a very small dataset for demo purpose
 
## Predict
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

- Basic usage: python predict.py <path_to_image_file>/file_name <path_to_checkpoint_file>/checkpoint.pth
- Options: 
  - __Return top  K most likely classes__: ```python predict.py <path_to_image_file>/file_name <path_to_checkpoint_file>/checkpoint.pth --top_k 3 ```
  - __Use a mapping of categories to real names__: ```python predict.py <path_to_image_file>/file_name <path_to_checkpoint_file>/checkpoint.pth --category_names cat_to_name.json ```
  - __Use GPU for inference__: ```python predict.py <path_to_image_file>/file_name <path_to_checkpoint_file>/checkpoint.pth --gpu ```


![image](https://user-images.githubusercontent.com/91800813/233997428-5e5e833d-cdb9-4e7d-96c2-11dd76f86801.png)

## Tech & Tool

__Language__: [Python](https://www.python.org/)

__Library__: [PyTorc](https://en.wikipedia.org/wiki/PyTorch)

__IDE__: [VS Code](https://code.visualstudio.com/)

__Formatting Tool__ :[Black](https://github.com/psf/black)
## Authors

- [Siddhartha Purwar](https://www.linkedin.com/in/siddp6/)

