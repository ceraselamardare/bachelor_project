#Bachelor degree project 
##Automatic description of human faces in portrait images

This project aims to obtain a textual description, based on a
portrait-type image, containing the gender, age, hair colour and the angle of orientation of the face of the person in the photograph.
This application was implemented by using four convolutional neural networks in parallel, one each 
for each important feature of the human face, using the Keras and TensorFlow libraries of the 
programming language.

###The project followed the steps:

1. Two datasets (IMDB-WIKI and CelebA) of considerable size were found, accompanied by corresponding labels. The first step in implementation of the application was to clean the database by removing images that did not contain a face 
human face. The images were then processed and the area of maximum interest, namely the human face, was cropped, using the face detection algorithm using the Haar feature-based cascaded classifiers from the OpenCv library.


2. The next step was to develop the four convolutional neural networks for each feature.Neural networks using transfer learning have been implemented for the estimation of gender, age and hair colour, which use weights from pre-trained models on very large datasets. These pre-trained models include MobileNet, MobiletNetV2, VGG19, ResNet.For the human face orientation angle, the approach was to develop from scratch a convolutional network that predicts the coordinates of the main key points of the face (eyes, tip of nose, chin and corners of mouth) and use these points together with the .solvePnP function in the OpenCV library. 


3. The trained models were then optimized by experimental trials varying 
hyperparameters and pre-trained models.
   
4. In order to demonstrate the functionality of the application, a graphical user interface has been implemented using the Tkinter library, through which the following can be tested 
any image. The interface allows the user to upload any portrait image from their personal photo collection 
and receive a textual description of it as a result.

![gui_final_crop](https://user-images.githubusercontent.com/93477871/139587791-f79261f4-1b90-4471-9dae-44358717b7ba.png)

   

###Datasets:
1. IMDB-WIKI https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
2. CelebA https://www.kaggle.com/jessicali9530/celeba-dataset
