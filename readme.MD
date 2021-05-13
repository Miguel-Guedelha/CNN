# CNN
Implementation of a Convolutional Neural Network for COMP530

The project itself has 2 main folders and .py files at the root of the project

## Dependencies

All dependencies are provided in the requirements.txt file for easy installation with:

* pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html

Some dependencies might be leftover and might not actually be used in the current final version of the project/implementation

## Embedding

The embedding used is available at https://fasttext.cc/docs/en/english-vectors.html (choose crawl-300d-2M.vec)
Once downloaded unzip it and place it in a folder at the root the of project called "embeddings"

## Folders

### Data

This folder contains the different datasets used for training and testing

### Models

This folder was supposed to contain pretrained models, however due to the file size limitations of GitHub, these pretained models were removed and the folder no longer exists in git. When main.py is run with all the different possible parameters, the trained models will be saved.

## Files in root folder

### cnn.py

Implementation of CNN

### main.py 

Main file for training and testing the neural network (instructions inside as comments)

### test_model.py 

Python program used to load a pretrained model previously obtained from main.py and see in detail the decisions made by the model.

* Confidence values above 50% lead to classifying as fake, and below lead to classifying as not fake
* Real value/label is shown next to confidence for checking accuracy of model

### utility.py

Python file with utility functions to reduce some of the complexity of main.py and test_model.py and remove code repetition where possible



