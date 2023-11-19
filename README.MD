# Music Selection App

## Overview

This application is designed to select music based on the atmosphere of a given photograph. It utilizes a pre-trained image classification model from TensorFlow Hub to predict the appropriate genre for the image, mapping it to a range of labels and associating songs with each genre.

## Usage

- **Upload Image:** Choose an image from your local folder using the "Upload Image" button.
- **Generate Random Image:** Generate a random noise image for testing using the "Generate Image" button.
- **Choose Random Image:** Choose a random image from the "pictures" folder.
- **Predict and Play Music:** Predict the image label, select the appropriate genre, and play a random song associated with that genre.
- **Play/Pause:** Control the music playback with the "Pause" and "Play" buttons.

## Prerequisites

- Python 3.x
- Required Python libraries (install using `pip install -r requirements.txt`):
  - tkinter
  - customtkinter
  - PIL
  - tensorflow
  - tensorflow_hub
  - numpy
  - pygame

## Program Results

The program predicts the image label, selects a music genre, and plays a random song based on the predicted genre.

## Conclusion

I have developed an application that utilizes a pre-trained image classification model to select music suitable for the atmosphere of a given photograph. The GUI is designed using the custom tkinter library for a more pleasant user experience. The application has a modular design, allowing for easy integration of new songs and adaptability to changes in the classification model.

This application can be further developed in the future to enhance its functionality. Potential ideas include searching for music on the internet, considering user preferences, and expanding the range of available songs.

## Author

Yehor Lashkul