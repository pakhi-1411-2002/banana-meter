# Banana Ripeness Detector

A computer vision web app that predicts the ripeness stage of a banana from an uploaded image and estimates how many days remain before it becomes overripe.

The app allows a user to upload a banana image, runs the image through a trained classification model, and displays the predicted ripeness stage along with a confidence score.

## Features

- Upload a banana image through a simple web interface
- Predict banana ripeness stage using a computer vision model
- Estimate approximate remaining shelf life
- Display model confidence using a visual confidence bar
- Give a simple user-friendly interpretation of the result

## Example Output

Prediction: Ripe  
Estimated time before overripe: 2–3 days  
Model confidence: 87%

## Tech Stack

- Python
- TensorFlow / Keras or PyTorch
- OpenCV
- NumPy
- Streamlit / Flask
- Matplotlib

## Project Structure

```text
banana-ripeness-detector/
│
├── app.py
├── model/
│   └── banana_model.h5
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   └── utils.py
├── assets/
│   └── sample_predictions/
├── requirements.txt
├── README.md
└── .gitignore
