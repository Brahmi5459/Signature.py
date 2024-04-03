import os
import cv2
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Function to load and preprocess images
def load_and_preprocess_data(data_path, label, augment=False):
    images = []
    labels = []

    for img_file in os.listdir(data_path):
        img_path = os.path.join(data_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100))  # Resize to (100, 100)
        images.append(img.flatten())  # Flatten the image into a 1D array
        labels.append(label)
        
        if augment:
            # Augmentation: horizontal flip
            img_flipped = cv2.flip(img, 1)
            images.append(img_flipped.flatten())
            labels.append(label)

    return images, labels

# Function to load and preprocess a single image
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))  # Resize to (100, 100)
    img_flat = img.flatten().reshape(1, -1)  # Reshape to a 2D array
    return img_flat

# Load genuine and fake signature data
genuine_images, genuine_labels = load_and_preprocess_data('C:\\Users\\Brahmi\\Downloads\\original dataset', label=0, augment=True)
fake_images, fake_labels = load_and_preprocess_data('C:\\Users\\Brahmi\\Downloads\\fraud dataset', label=1)

# Combine genuine and fake data
all_images = genuine_images + fake_images
all_labels = genuine_labels + fake_labels

# Convert to numpy arrays
X = np.array(all_images)
y = np.array(all_labels)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X, y)

best_params = grid_search.best_params_

# Use the best hyperparameters for the final model
clf = RandomForestClassifier(**best_params, random_state=42)
clf.fit(X, y)

# Streamlit app
st.title("Signature Detection App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))

    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_flat = image.flatten().reshape(1, -1)
    prediction = clf.predict(img_flat)

    if prediction[0] == 0:
        st.write("The signature is genuine.")
    elif prediction[0] == 1:
        st.write("The signature is fake.")
