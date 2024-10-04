# Install necessary libraries
!pip install opencv-python-headless
!pip install scikit-learn
!pip install tensorflow

import os
import cv2
import numpy as np
import shutil
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from google.colab import files
from zipfile import ZipFile

# Create a folder for your output
output_base_dir = '/content/player_classification_output'
player_dirs = [os.path.join(output_base_dir, f'player_{i}') for i in range(1, 5)]
for player_dir in player_dirs:
    os.makedirs(player_dir, exist_ok=True)

# Load the pretrained MobileNetV2 model
mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Function to extract deep features using MobileNetV2
def extract_deep_features(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = mobilenet.predict(image)
    return features.flatten()

# Load images and extract features
def load_and_extract_features(image_paths):
    features = []
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            deep_features = extract_deep_features(img)
            features.append(deep_features)
            images.append((img_path, img))
    return features, images

# Dimensionality reduction with PCA
def reduce_dimensions(features):
    pca = PCA(n_components=50)  # Reduce to 50 dimensions
    reduced_features = pca.fit_transform(features)
    return reduced_features

# Clustering function using DBSCAN
def cluster_images(features):
    # Adjust eps and min_samples for DBSCAN based on your data
    clustering = DBSCAN(eps=0.5, min_samples=2)
    labels = clustering.fit_predict(features)
    return labels

# Save clustered images to output folders
def save_clustered_images(labels, images):
    unique_labels = set(labels)
    for i, (label, (img_path, _)) in enumerate(zip(labels, images)):
        if label != -1:  # Ignore noise points
            output_dir = player_dirs[label]  # Map cluster labels directly to player folders
            shutil.copy(img_path, os.path.join(output_dir, f'image_{i}.jpg'))

# Upload your zip file containing the images
uploaded = files.upload()

# Extract the zip file
for filename in uploaded.keys():
    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('/content')

# Specify the directories for the images
top_dir = '/content/two_players_top'
bot_dir = '/content/two_players_bot'

# Process images from top and bottom folders
top_image_paths = [os.path.join(top_dir, img) for img in os.listdir(top_dir) if img.endswith('.jpg')]
bot_image_paths = [os.path.join(bot_dir, img) for img in os.listdir(bot_dir) if img.endswith('.jpg')]

# Extract features for top and bottom player images
top_features, top_images = load_and_extract_features(top_image_paths)
bot_features, bot_images = load_and_extract_features(bot_image_paths)

# Combine the top and bottom features for clustering (since we have 4 players in total)
all_features = top_features + bot_features
all_images = top_images + bot_images

# Reduce dimensions for clustering
reduced_features = reduce_dimensions(all_features)

# Apply DBSCAN clustering
labels = cluster_images(reduced_features)

# Save images to respective player folders
save_clustered_images(labels, all_images)

# Zip the output folder and allow download
shutil.make_archive('/content/player_classification_output', 'zip', '/content/player_classification_output')
files.download('/content/player_classification_output.zip')
