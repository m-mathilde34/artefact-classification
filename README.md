# How To Run The Code

## Downloading The Images (downloadImg.py)
First, create a folder which will hold your dataset such as "dataset".
Then create the subfolders for each of your classes such as :\
1_egyptian\
2_grecoroman\
3_orient\
4_islam\
5_byzantium\
6_asian

Then, verify that the following variables are correct :

   1. filepath : the absolute filepath of the class for which you are downloading the images
   2. filename : a String which will be used as the beginning of a filename for an image. This will be the same for every images in a given class.
We followed the convention "CLASSID_MUSEUM_UNIQUEID". In this case only "UNIQUEID" varies per image. For example, the egyptian class filename would be : "1_LOUVRE_"
   3. Line 8, add the name of the CSV file which you would like to download the images from. This code is meant for CSV files containing the images' URL ONLY.

## Preprocessing The Data

1. __Resizing the images__ (preprocessing_size.py)\
Create a folder which will hold your resized dataset.
Verify that the following variables are correct, then run the code :
   1. input_folder : the name of the folder holding your raw dataset
   2. output_folder : the name of the folder you created to hold your resized dataset
***
2. __Getting rid of blank images__ (tidy_dataset.py)\
Before running this code, make sure you find a blank image in the dataset and extract it from it.
Place it outside your dataset and copy the filepath to it. Paste the copied string in the img1 variable.\
Create a folder outside of the dataset which will hold all the found blank images (this is to ensure we do not delete
good data by mistake).\
Verify that the following variables are correct, then run the code :
   1. input_folder : the name of the folder holding your resized dataset
   2. img1 : the name of an empty picture manually found in the dataset, now placed outside of it
   3. line 50, change "D:\empty_pictures" to the name of the folder which will hold all the blank images found
*** 
3. __Creating a Test set__ (train_test_split.py)\
Create a folder which will hold your test data per class.\
Verify that the following variables are correct, then run the code :
   1. source_dir : the name of the folder holding your resized dataset
   2. test_dir : the name of the folder you created to hold your test data

## Running The Models

For faster and more efficient results, we used Docker to run the models.
* Open Docker Desktop
* In the Dockerfile_train, under " # Execute CNN ", select whichever model you want to run by commenting out the one you do not wish to use.
  * cnn_artefact_classification.py is for ResNet50.
  * cnn_efficientnet_model.py is for the EfficientNetB0-2 models. Depending on which version you want to run (B0, B1, or B2), you will need to change this in the code where appropriate.
* In your terminal, create the image using the following line : *docker build -t train_model -f Dockerfile_train .*
* Run the models in the terminal using the following : docker run --gpus all -it -v *project_folder_path*:/app/ -v *image_folder_path*:/app/data -it train_model
  * *project_folder_path* : the path in which this project is stored. Make sure to add __:/app/__ straight after it
  * *image_folder_path* : the path to the dataset. Make sure to add __:/app/data__ straight after it. 
* Between each model evaluation, adapt the following:
  * Epoch file name in *save_history()* 
  * Model file name line 164
  * Parameters such as optimizer if needed

## Evaluating The Models (evaluating_model.py)

Once again, use Docker to run this file to use the GPU.
* Open Docker Desktop
* In the Dockerfile_evaluate, under " # Execute CNN ", select whichever file you want to run by commenting out the one you do not wish to use.
  * evaluating_model.py is to evaluate the models as required here.
  * model_misclassified.py is to use when trying to visualise misclassified images, as seen later on.
* In your terminal, create the image using the following line : *docker build -t evaluate_model -f Dockerfile_evaluate .*
* Run the model evaluation in the terminal using the following : docker run --gpus all -it -v *project_folder_path*:/app/ -v *image_folder_path*:/app/data -it evaluate_model
  * *project_folder_path* : the path in which this project is stored. Make sure to add __:/app/__ straight after it
  * *image_folder_path* : the path to the dataset. Make sure to add __:/app/data__ straight after it.

For each model you want to evaluate, simply change the model path in the load_model() function.

## Getting the misclassified images (model_misclassified.py)

Outside of your dataset folder, create a folder which will hold all of our misclassified data.\
Once again, use Docker to run this file to use the GPU.
* Open Docker Desktop
* In the Dockerfile_evaluate, under " # Execute CNN ", select whichever file you want to run by commenting out the one you do not wish to use.
  * evaluating_model.py is to evaluate the models as seen above.
  * model_misclassified.py is to use in this case, when trying to visualise misclassified images.
* In your terminal, create the image using the following line : *docker build -t misclassified_model -f Dockerfile_evaluate .*
* Run the model evaluation in the terminal using the following : docker run --gpus all -it -v *project_folder_path*:/app/ -v *image_folder_path*:/app/data -it misclassified_model
  * *project_folder_path* : the path in which this project is stored. Make sure to add __:/app/__ straight after it
  * *image_folder_path* : the path to the dataset. Make sure to add __:/app/data__ straight after it.

For each model you want to do this for, simply change the model path in the load_model() function.

## Using Data Balancing Techniques

1. __Class Weights__\
Repeat the steps seen above to run your model, with the following modification:
   * Uncomment methods get_image_labels() and get_classes_weights()
   * Uncomment lines 100, 102, and 153

To evaluate this model, use the same steps as detailed in the section above 'Evaluating The Models'
***

2. __Undersampling__ (undersampling.py)\
In order to keep your original dataset intact, create a new folder and call it "undersampling" for example.\
Verify that the following variables are correct, then run the code:
   1. source_dir : the name of the folder holding your dataset
   2. target_dir : the name of your newly created folder which will hold the undersampled dataset
   3. number_to_move : the number of images you will want each folder to have. In our case the size of the smallest class

***
3. __Oversampling__ (data_augmentation.py)\
If you would like to keep the original set of images of your smallest class kept intact, prior to running this code do the following:
   * Move the folder outside of your dataset
   * In your dataset, create a new and empty folder and call it exactly the same as the one you removed with the added "_augmented"

Verify that the following variables are correct, then run the code :
* images_folder : the name of the folder holding your smallest class
* line 43, add the path to your new folder where your augmented images will be stored

You can then copy all images of your original folder (the non-augmented images) and add them to your augmented folder by pasting them into it.
This way you can keep your original data intact should you need to use it further, and you can test this data balancing technique.
