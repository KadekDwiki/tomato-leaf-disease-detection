import os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump

# 1. Load the dataset
def load_data(data_dir):
   images = []
   labels = []
   classes = os.listdir(data_dir)
   class_dict = {class_name: idx for idx, class_name in enumerate(classes)}
   
   for class_name in classes:
      class_path = os.path.join(data_dir, class_name)
      for img_name in os.listdir(class_path):
         img_path = os.path.join(class_path, img_name)
         img = cv2.imread(img_path)
         img = cv2.resize(img, (128, 128))  # Resize for uniformity
         images.append(img)
         labels.append(class_dict[class_name])
   
   return np.array(images), np.array(labels), class_dict

# 2. Feature Extraction using Color Histogram
def extract_features(images):
   features = []
   for img in images:
      hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
      hist = cv2.normalize(hist, hist).flatten()
      features.append(hist)
   return np.array(features)

# 3. Main training pipeline
data_dir = "D:/Python dev/tomato-leaf-disease-detection/datasets"  # Replace with your dataset path
images, labels, class_dict = load_data(data_dir)
print(f"Loaded {len(images)} images from {len(class_dict)} classes.")

# Extract features
features = extract_features(images)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train LinearSVC model
print("Training the model...")
model = make_pipeline(StandardScaler(), LinearSVC(random_state=42, max_iter=10000))
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model_path = "tomato_disease_model.joblib"
dump(model, model_path)
print(f"Model saved to {model_path}")

# Save class dictionary
class_dict_path = "class_dict.npy"
np.save(class_dict_path, class_dict)
print(f"Class dictionary saved to {class_dict_path}")
