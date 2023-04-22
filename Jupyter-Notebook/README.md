
# FOR Jupyter Notebook

# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img width="260" alt="output_36_0" src="https://user-images.githubusercontent.com/91800813/233777768-171b8cd2-2636-44a4-98e7-66d695d6d609.png">
The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.


```python
import os
import json
import torch

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models
from PIL import Image 
```


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```


```python
# Hyperparmeters
batch_size = 64
epochs = 2
learning_rate = 0.001
criterion = nn.NLLLoss()

# Other constants
input_size = 25088
output_size = 102
hidden_layers = 512
drop_rate = 0.2

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
```


```python
normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406], 
    std = [0.229, 0.224, 0.225]
)

data_transforms = {}

data_transforms['train'] = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    normalize
    ])

data_transforms['val'] = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
    ])

data_transforms['test'] = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
    ])

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
    'val': datasets.ImageFolder(valid_dir, transform = data_transforms['val']),
    'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])
} 

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size, shuffle=True),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size, shuffle=True),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
}
```

### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

<font color='red'>**Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.</font>


```python
model = models.vgg16(pretrained=True)
```

    Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.torch/models/vgg16-397923af.pth
    100%|██████████| 553433881/553433881 [00:04<00:00, 112528847.78it/s]



```python
for param in model.parameters():
    param.requires_grad = False
```


```python
class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_units, drop_rate, output_size):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units // 2)
        self.layer3 = nn.Linear(hidden_units // 2, output_size)

        self.dropout = nn.Dropout(drop_rate)

        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.layer2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.layer3(out)

        out = self.output(out)

        return out
```


```python
model.classifier = MyNetwork(
    input_size, hidden_layers, drop_rate, output_size
)
```


```python
def train_network():
    trainloader = dataloaders["train"]
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    steps = 0

    for epoch in range(epochs):
        training_loss = 0
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            Loss = criterion(output, labels)

            Loss.backward()
            optimizer.step()

            training_loss += Loss.item()

        model.eval()
        with torch.no_grad():  # so that gradient do not get re-calculate
            loss, accuracy = get_train_validation()

        print(
            "Epoch: {}/{}\nTraining Loss: {:.4f}\nValidation Loss: {:.4f}\nvalidation  Accuracy: {:.3f}%\n\n".format(
                epoch + 1,
                epochs,
                training_loss,
                loss / len(dataloaders["val"]),
                accuracy / len(dataloaders["val"]) * 100,
            )
        )

        model.train()

    return model
```


```python
def get_train_validation():
    loss, accuracy = 0, 0
    model.to(device)
    validloader = dataloaders["val"]

    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)  # Generate predictions
        loss += criterion(output, labels).item()  # Calculate loss
        ps = torch.exp(output)
        equity = labels == ps.max(dim=1)[1]
        accuracy += equity.type(torch.FloatTensor).mean()

    return loss, accuracy
```


```python
trained_model = train_network()
```

    Epoch: 1/2
    Training Loss: 282.7597
    Validation Loss: 1.1246
    validation  Accuracy: 69.774%
    
    
    Epoch: 2/2
    Training Loss: 138.5618
    Validation Loss: 0.6321
    validation  Accuracy: 82.481%
    
    


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.


```python
def get_test_validation():
    accuracy, data_size = 0, 0

    model.to(device)

    model.eval()
    for images, labels in dataloaders['test']:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        ps = torch.exp(output)
        equity = labels == ps.max(dim=1)[1]
        data_size += labels.size(0)
        accuracy += equity.type(torch.FloatTensor).sum().item()

    return accuracy, data_size
```


```python
with torch.no_grad():
    accuracy, total = get_test_validation()
print('Accuracy: {}%\n'.format(100 * accuracy / total))

```

    Accuracy: 80.34188034188034%
    


## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
def save_model(save_dir):
    checkpoint = {
        "architecture": 'vgg16',
        "state_dict": model.state_dict(),
        "class_to_idx": image_datasets['train'].class_to_idx,
        'model': models.vgg16(pretrained=True),
        'classifier': model.classifier
    }


    torch.save(checkpoint, save_dir)
```


```python
save_model('checkpoint.pth')
```

    /opt/conda/lib/python3.6/site-packages/torch/serialization.py:193: UserWarning: Couldn't retrieve source code for container of type MyNetwork. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "


## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python
def get_loaded_model(checkpoint):
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False

    return model
```


```python
model = get_loaded_model('checkpoint.pth') 
```

# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python
def process_image(image):
    image = Image.open(image)

    image_transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]
    )

    return image_transform(image)
```


```python
def imshow(image, title=None):
    fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (std * image) + mean
    image = np.clip(image, 0, 1)
    if title:
        ax.set_title(title)
    ax.imshow(image)
```


```python
imshow(process_image('flowers/test/3/image_06641.jpg'))
```


<img width="260" alt="output_30_0" src="https://user-images.githubusercontent.com/91800813/233777732-f606da7c-82c1-40c2-a429-b0c9552dd0de.png">


## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
 def get_prediction(image_path, model, device, topk=5):
    image = process_image(image_path).unsqueeze(0).float()
    model, image = model.to(device), image.to(device)
    model.eval()
    
    with torch.no_grad():
        results = torch.exp(model.forward(image))
        result_top_k = results.topk(topk)
        
        probs, classes = result_top_k[0].data.cpu().numpy()[0], result_top_k[1].data.cpu().numpy()[0]
        
        idx_to_class = {key: value for value, key in model.class_to_idx.items()}
        classes = [idx_to_class[classes[i]] for i in range(classes.size)]

    return probs, classes
```


```python
probs, predict_classes = get_prediction(
    'flowers/test/3/image_06641.jpg', model, device
)

print(predict_classes)
print(probs)

```

    ['3', '88', '95', '4', '72']
    [ 0.24047619  0.20982844  0.12638119  0.12139541  0.03391226]


## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='assets/inference_example.png' width=300px>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
def view_classify(image_path, probs, classes):
    name = [cat_to_name[i] for i in classes]
    imshow(process_image('flowers/test/3/image_06641.jpg'), cat_to_name[classes[0]])
    
    _, (ax_img, ax_gph) = plt.subplots(nrows=2)
    
    ax_img.axis('off')
    
    ax_gph.barh(np.arange(len(name)), probs)
    
    ax_gph.set_yticks(np.arange(len(name)))
    ax_gph.set_yticklabels(name)
    
    ax_gph.invert_yaxis()
```


```python
view_classify('flowers/test/3/image_06641.jpg', probs, predict_classes)
```


<img width="260" alt="output_36_0" src="https://user-images.githubusercontent.com/91800813/233777740-1368491d-33d6-40b9-aa35-eea45014e649.png">



<img width="446" alt="output_36_1" src="https://user-images.githubusercontent.com/91800813/233777745-229473b4-5487-4a5f-8b8a-9025c4ae41dc.png">


<font color='red'>**Reminder for Workspace users:** If your network becomes very large when saved as a checkpoint, there might be issues with saving backups in your workspace. You should reduce the size of your hidden layers and train again. 
    
We strongly encourage you to delete these large interim files and directories before navigating to another page or closing the browser tab.</font>


```python
os.remove("checkpoint.pth")
```
