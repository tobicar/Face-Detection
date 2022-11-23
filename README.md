# Face-Detection

Face Detection with deep neural networks

# Installation:
- link to github repository: https://github.com/tobicar/Face-Detection
- created with pycharm IDE
- parts of the code better executable with extension [pycharm cell mode](https://plugins.jetbrains.com/plugin/7858-pycharm-cell-mode)



# Second Milestone:
goal --> Detection of whether a face is present in the image or not. The prediction indicates the probability of a face.

## Selection of the database:

1. class (pictures containing present face):
- dataset [UTKFace](https://susanqq.github.io/UTKFace/)
2. class (pictures not containing faces, especially animals):
- [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- [10-monkey-species](https://www.kaggle.com/datasets/slothkong/10-monkey-species)
- [landscape-pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)
- [natural-images](https://www.kaggle.com/datasets/prasunroy/natural-images)

The dataset was selected to distinguish between humans and animals in particular. 
The dataset contains of 15.523 pictures with 6.941 faces and 8.582 no faces.

## structure of the project:

### python files:
- [helper.py](helper.py):
  - file with functions for:
    - loading and saving models
    - import train and test images
    - compile and train models
    - generate train histories
    - predict images
- [trainingScratch.py](trainingScratch.py):
  - file for train different models with different hyperparameters from scratch
  - the trained models are automatically saved to the directory [saved_model](saved_model)
  - history of the training process is saved to the directory [plots](plots)
- [trainingTransfer.py](trainingTransfer.py):
  - file for automatically train different models and hyperparameters with transfer learning
  - the trained models are automatically saved to the directory [saved_model](saved_model)
  - history of the training process is saved to the directory [plots](plots)
- [initialTestData.py](initialTestData.py):
  - splits the raw data from the folder [rawdata](images/rawdata) into a training- and testset
  - 15 % of the pictures are randomly saved in the [test](images/test) and 85 % in the [train](images/train)
  - to do this the project must contain the following file structure of pictures:
    - images:
      - rawdata (folder with the raw dataset)
        - face
        - no_face
      - test:
        - face
        - no_face
      - train:
        - face
        - no_face
- [predictimage.py](predictimage.py):
  - opens a file dialog where you can choose a image to do a prediction with
  - To change the model for prediction change the variable "PATH_TO_MODEL" to the corresponding model saved in [saved_model](saved_model)
  - after choosing a image the image will be displayed and the prediction is shown under the picture
- [generatePlotsTensorboard.py](generatePlotsTensorboard.py):
  - generate plots for the presentation from csv-training data, which can be exported from the tensorboard dashboard
  - to generate a plot change the file paths for the pd.read_csv
- [evaluation.py](evaluation.py):
  - evaluates specific models on the test datset which is specified in the [images/test](images/test) directory
  - to change evaluated models change "model_path.__contains__()"
  - the results are stored in a csv file with the current timestamp in the [evaluation](evaluation) directory
- [convertToCoreML](convertToCoreML.py):
  - file converts a specific model to a core ML file for prediction on a iphone
  - with PATH_TO_MODEL the desired model can be choosen

### directories:
- [evaluation](evaluation)
  - contains csv files with evaluated models on the test dataset
- [images](images)
  - contains the train and test images
- [logs](logs)
  - automatically generated by tensorboard
- [plots](plots)
  - contains plots of the different training runs, which are automatically generate during training
- [presentation](presentation)
  - contains plots and csv files that are shown in the presentation
- [saved_model](saved_model)
  - contains all trained models, that can be loaded


# Third Milestone:

Aufgabe: Erkennung von Merkmalen

Implementieren Sie eine Methode zur Erkennung von Merkmalen basierend auf einem Gesichtsbild
Merkmale:
- Maskenerkennung (Klassifikation: Maske/ keine Maske) -> binary Cross entropy
- Altersbestimmung (Regression) -> MSE
- Gesichtserkennung aus Meilenstein2 (transferLearning Model) -> binary Cross entropy

Evaluation: Visualisierung und Quantitative Bewertung der Ergebnisse


Bild Ordner Struktur:

Face (14.000)
- Mask (7.000) -> augmentierung
- No Mask (7.000)
    - Age
No_face (8.500) -> (14.000)


Filename, image path, face (0/1), mask (0/1), age (int für nicht vorhanden -1) 


Erkenntnisse:
- Es gibt viele Maskenbilder mit asiatischen Gesichtern




# Fourth Milestone:
