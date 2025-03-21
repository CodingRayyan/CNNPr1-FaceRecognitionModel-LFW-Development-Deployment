import streamlit as st
import tensorflow as tf  # ✅ Keep it as 'tensorflow', even if installed as 'tensorflow-cpu'
import numpy as np
from PIL import Image

# Load the saved model
@st.cache_resource  # Caches the model to avoid reloading on every run
def load_model():
    return tf.keras.models.load_model('lfw_prone.keras')

model = load_model()

# CIFAR-10 class labels
class_labels = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", 
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Streamlit UI
st.title("LFW Face Recognition")
st.write("Upload an image and the model will predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Process image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((32, 32))  # Resize to CIFAR-10 input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = class_labels[np.argmax(predictions)]

    # Show result
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: **{predicted_class}**")
