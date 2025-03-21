![image](https://github.com/user-attachments/assets/e51fc620-c69f-4cf2-92ca-35ca3dd0264b)

# CNN Project01: Face Recognition using LFW DataSet

## Overview
This project implements a **Convolutional Neural Network (CNN)** for **Face Recognition** using deep learning. The model is trained to recognize and classify faces from an image dataset.

## Features
- Face detection and preprocessing
- CNN-based feature extraction
- Model training and evaluation
- Real-time face recognition using OpenCV

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib (for visualization)

## Dataset
- You can use a dataset like **Labeled Faces in the Wild (LFW)** or create a custom dataset.
- Ensure dataset is structured as:
  ```
  dataset/
    ├── train/
    │   ├── person_1/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   ├── person_2/
    │   │   ├── img1.jpg
    ├── test/
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cnn-face-recognition.git
   cd cnn-face-recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Training
1. Preprocess the dataset (resize, normalize images).
2. Train the CNN model using TensorFlow/Keras:
   ```python
   python train.py
   ```
3. Save the trained model.

## Face Recognition (Testing)
- To test the model on new images:
  ```python
  python recognize.py --image test_image.jpg
  ```

## Future Improvements
- Improve accuracy with deeper CNN architectures
- Implement Transfer Learning (e.g., using VGG16, ResNet)
- Add real-time face detection with OpenCV

## License
This project is licensed under the **MIT License**.

