# **BDD100K Pedestrian Dataset Preparation and Detection for Autonomous Driving**

This guide walks you through the **first stage** of preparing the **BDD100K dataset** specifically for **pedestrian detection**, an essential task in **autonomous driving systems**. The goal of this step is to create a balanced and properly cleaned dataset with images containing visible pedestrians, ready for model training.

By following this guide, you will create a dataset optimized for training pedestrian detection models, contributing to the **accuracy**, **reliability**, and **safety** of autonomous vehicle systems.

---

## 📦 About the BDD100K Dataset

The **BDD100K** dataset, developed by **Berkeley DeepDrive**, is one of the largest and most diverse datasets available for autonomous driving research. It contains:

* **100,000 high-resolution images** from a wide variety of driving scenarios, captured in real-world settings across different cities, times of day, and weather conditions.
* Rich annotations, including:

  * **Pedestrians** (humans)
  * **Vehicles**
  * **Traffic signs**
  * **Lane markings**
  * **Cyclists**
* **Metadata** such as weather, time of day, and scene type for each image.
* The dataset is split into:

  * `train` (for model training)
  * `val` (for validation)
  * `test` (for evaluation).

---

## 🎯 Objective

The main objective of this guide is to prepare a subset of the **BDD100K dataset** focused on **pedestrian detection** by:

* Extracting images that **contain pedestrians** and ensuring the people are **clearly visible** in the images.
* Removing or filtering out images with **poorly visible pedestrians**, where they occupy less than a certain threshold of the image area.
* Including a balanced number of **negative samples**, i.e., images that do **not** contain pedestrians, for effective detection training.

---

## 🔍 Why Pedestrian Detection Matters

Pedestrian detection is a critical task for autonomous vehicles, as pedestrians are among the most vulnerable road users and pose a significant risk to safety. The importance of accurate pedestrian detection cannot be overstated:

* **Safety and Ethics**: Pedestrians are unpredictable and often difficult for autonomous systems to track. Ensuring that an autonomous vehicle can detect pedestrians at all times is a matter of **safety** and **ethical responsibility**.
* **Avoiding Collisions**: Autonomous vehicles must be able to detect and react to pedestrians quickly to avoid potential accidents.
* **Improving Vehicle Intelligence**: Pedestrian detection contributes to the overall **intelligence** of an autonomous system, enabling better decision-making in complex environments.

---

## 🔗 Dataset Download Links

To begin, download the necessary files from the provided links:

* 🖼️ **Images (100K)**:
  [bdd100k\_images\_100k.zip](http://128.32.162.150/bdd100k/bdd100k_images_100k.zip)

* 🏷️ **Labels**:
  [bdd100k\_labels.zip](http://128.32.162.150/bdd100k/bdd100k_labels.zip)

The dataset is organized into three directories:

```
- train/
- val/
- test/
```

These folders align with both the **images** and **annotations** for easy use.

---

## 📁 Recommended Project Structure

Once the datasets are unzipped, organize your project directory for easier access and management:

```
project_root/
├── data/
│   ├── images/train/val/test/
│   └── labels/train/val/test/
├── bdd_people/
│   ├── with_people/train/val/test/
│   └── without_people/train/val/test/
├── get_dataset_persons.py
└── Pedestrian Detection using SVM and HOG.ipynb

```

* `data/images/` and `data/labels/`: The locations for your original dataset.
* `bdd_people/with_people/`: Images containing pedestrians.
* `bdd_people/without_people/`: Images without pedestrians.
* `get_dataset_persons.py`: The script that handles the filtering and organization of the dataset.

---

## ⚙️ Script: `get_dataset_persons.py`

### Script Overview

The **`get_dataset_persons.py`** script is designed to:

* **Parse JSON annotation files** from the BDD100K dataset.
* **Filter images** based on the presence of pedestrians, using bounding boxes labeled `"person"`.
* Apply a **size threshold** to ensure that pedestrians are large enough in the image for meaningful detection (e.g., at least 0.5% of the total image area).
* Organize the resulting images into two categories:

  * **With People**: Images that contain pedestrians.
  * **Without People**: Images that do not contain pedestrians.
* **Balance the dataset** by ensuring that both categories (with and without people) have an equal number of samples, which is important for training a reliable model.

### Detailed Workflow

1. **Load and Parse Labels**: The script reads the JSON files in the `labels` folder, extracts the `"person"` annotations, and checks whether they meet the visibility threshold.
2. **Filter by Area**: The script filters images based on the **bounding box size** for pedestrians, ensuring that only sufficiently large pedestrians are kept.
3. **Save the Filtered Data**: The images are saved into the `bdd_people/` directory, split by `train`, `val`, and `test` datasets.

---

## 🛠️ Usage

To run the script, execute the following command from the project root directory:

```bash
python get_dataset_persons.py
```

### Optional Arguments:

| Argument           | Description                                    | Default        |
| ------------------ | ---------------------------------------------- | -------------- |
| `--subset`         | Dataset split to process (train/val/test)      | `train`        |
| `--size_threshold` | Minimum person bounding box area as % of image | `0.005` (0.5%) |

### Example:

```bash
python get_dataset_persons.py --subset val --size_threshold 0.01
```

This command will process the **validation** split (`val`) and will only include pedestrians with bounding boxes that occupy at least 1% of the image area.

---

## 📤 Output

The script will generate the following directory structure:

```
bdd_people/
├── with_people/
│   └── train/ (or val/test)
└── without_people/
    └── train/ (or val/test)
```

* The images in `with_people/` contain clearly visible pedestrians.
* The images in `without_people/` contain no pedestrians, which are useful as **negative samples** for training.

The filtered dataset is now ready for use in **object detection models** such as YOLO, Faster R-CNN, or SSD.

---

## 🧠 Why This Step is Critical

Pedestrian detection plays a vital role in the **autonomous driving pipeline**, as it ensures that the system can:

* **Avoid collisions** with pedestrians, which is the primary goal for safety.
* **Predict and react to pedestrian movements** in unpredictable urban environments.
* **Improve the vehicle's ability** to navigate in real-world conditions, including crowded streets, crosswalks, and urban centers.

Poorly prepared datasets can lead to:

* **Biases** in detection models, causing them to either miss pedestrians or produce false positives.
* **Low model accuracy** due to small or poorly annotated pedestrian instances.
* **Safety concerns** if the model fails to recognize pedestrians correctly.

By ensuring clean, visible annotations, this dataset preparation step helps improve the model’s **accuracy**, **robustness**, and **safety** in real-world scenarios.

---


# 🚶‍♂️ Pedestrian Detection using SVM and HOG

This project leverages **HOG (Histogram of Oriented Gradients)** for feature extraction and **SVM (Support Vector Machine)** for classifying images into two categories: images with pedestrians and images without pedestrians. The combination of these two techniques allows for efficient and effective pedestrian detection, which is crucial for applications in autonomous driving and computer vision. (In order to continue with the analysis, it is necessary to first create datasets from the previous steps and have downloaded the information.)

## 📦 Required Libraries

Before running the script, make sure to install the necessary dependencies. This includes libraries for image processing and machine learning:

```bash
pip install scikit-learn scikit-image
```
* **Python 3.7+**
* **Pillow** for image processing
* **NumPy** for handling arrays
* **Matplotlib** for visualizations
* **Seaborn** for enhanced plotting
* **scikit-image** for image processing, HOG, and resizing
* **scikit-learn** for machine learning and SVM

## 📊 **What is HOG?**

**HOG** (Histogram of Oriented Gradients) is a feature extraction technique commonly used in image processing for object detection. It works by calculating the gradient and orientation of edges in small, localized regions of the image, allowing it to capture the shapes and textures of objects, such as pedestrians. The process is broken down into these steps:

1. **Gradient Calculation**: Identifying changes in intensity to detect edges.
2. **Cell Division**: Dividing the image into small cells to analyze local structures.
3. **Block Normalization**: Grouping cells into blocks and normalizing the histograms to improve performance under varying lighting conditions.
4. **Feature Vector**: Combining the histograms into a single feature vector representing the image.

## 💻 **What is SVM?**

**SVM** (Support Vector Machine) is a powerful supervised machine learning algorithm used for classification. It works by finding the hyperplane that best separates the data points of different classes (in this case, with pedestrians and without pedestrians). The goal is to maximize the margin between the two classes, making it highly effective even with high-dimensional data, such as image features extracted by HOG.

The SVM classifier is trained on a set of positive (images with pedestrians) and negative (images without pedestrians) samples, then used to predict whether new images contain pedestrians.

## 🧠 **How it Works in This Notebook**

In the `Pedestrian Detection using SVM and HOG.ipynb` notebook, we go through the following steps:

1. **Feature Extraction**: We extract HOG features from a set of training images (both positive and negative samples).
2. **Training the Model**: We use an SVM classifier to learn the differences between images with pedestrians and those without.
3. **Evaluation**: The model is evaluated on a test set to check its accuracy and performance.

The notebook is designed to be straightforward and easy to follow, with comments and code snippets to guide you through each step.


## 📸 Load Images and Assign Labels

1. Define the paths for images with and without pedestrians.
2. Load up to 800 images per class to speed up processing.

```python
# Paths for images with and without pedestrians
path_with_pedestrians = "bdd_people/with_people/train/*.jpg"
path_without_pedestrians = "bdd_people/without_people/train/*.jpg"

# Load up to 800 images per class
images_with_people = glob.glob(path_with_pedestrians)[:800]
images_without_people = glob.glob(path_without_pedestrians)[:800]

# Combine all image paths and labels
all_images = images_with_people + images_without_people
all_labels = [1] * len(images_with_people) + [0] * len(images_without_people)  # 1 = pedestrian, 0 = no pedestrian

print(f"🔍 Images with pedestrians: {len(images_with_people)}")
print(f"🔍 Images without pedestrians: {len(images_without_people)}")
```

## 🧑‍🏫 Split Data into Training and Test Sets

We use 80% of the data for training and 20% for testing.

```python
# Split image paths into training and test sets
x_train_paths, x_test_paths, y_train_raw, y_test_raw = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42
)
```

## 🧑‍🔬 Extract HOG Features from Images

The `extract_hog` function takes an image path, converts it to grayscale, resizes it, and then extracts the HOG features.

```python
# Function to extract HOG features
def extract_hog(image_path, size=(128, 128)):
    image = imread(image_path)
    gray_image = rgb2gray(image)
    resized_image = resize(gray_image, size)
    features, _ = hog(resized_image,
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=True)
    return features

# Extract HOG features for training and test sets
X_train = [extract_hog(path) for path in tqdm(x_train_paths, desc="🔨 Extracting HOG (Train)", disable=True)]
X_test = [extract_hog(path) for path in tqdm(x_test_paths, desc="🔍 Extracting HOG (Test)", disable=True)]

# Convert labels and features to NumPy arrays
y_train = np.array(y_train_raw)
y_test = np.array(y_test_raw)

X_train = np.vstack(X_train)
X_test = np.vstack(X_test)
```

## 👀 Visualize HOG Features of Example Images

Here we visualize the HOG features of one pedestrian and one non-pedestrian example.

```python
# Function to return both the grayscale and HOG image
def visualize_hog(image_path):
    image = imread(image_path, as_gray=True)
    image = resize(image, (90, 90))  # Resize for better visualization
    _, hog_image = hog(image, orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys',
                       visualize=True)
    return image, hog_image

# Visualize one pedestrian and one non-pedestrian example
example_with = images_with_people[0]
example_without = images_without_people[0]
img1, hog1 = visualize_hog(example_with)
img2, hog2 = visualize_hog(example_without)

# Display original and HOG-transformed images
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(img1, cmap='gray'); axs[0, 0].set_title("Original (with pedestrian)")
axs[0, 1].imshow(hog1, cmap='gray'); axs[0, 1].set_title("HOG (with pedestrian)")
axs[1, 0].imshow(img2, cmap='gray'); axs[1, 0].set_title("Original (without pedestrian)")
axs[1, 1].imshow(hog2, cmap='gray'); axs[1, 1].set_title("HOG (without pedestrian)")
for ax in axs.ravel(): ax.axis('off')
plt.tight_layout(); plt.show()
```

## ⚙️ Train an SVM Model Using GridSearchCV

We use **GridSearchCV** to search for the best hyperparameters for the SVM model.

```python
# Define a hyperparameter grid to search over
param_grid = {
    'C': [0.1, 1, 10],         # Regularization parameter
    'gamma': [0.01, 0.1, 1],   # RBF kernel coefficient
    'kernel': ['rbf']          # Use RBF kernel
}

# Perform grid search with cross-validation
print("🔧 Searching for best hyperparameters...")
grid_search = GridSearchCV(SVC(), param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
```

## 💡 Display the Best Model Configuration

After grid search completes, the best hyperparameters will be displayed.

```python
# Display the best model configuration
best_model = grid_search.best_estimator_
print("\n✅ Best hyperparameter combination:")
print(grid_search.best_params_)
```

## 📊 Display Classification Results

We use the best model to make predictions on the test set and visualize the results using a confusion matrix and classification report.

```python
# Predict the test set labels
y_pred = best_model.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Display classification metrics
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Define class names
class_names = ['No pedestrian', 'Pedestrian']
```

## 📸 Visualize Predictions on Test Images

Finally, we visualize the predictions on the test images with color-coded labels.

```python
# Visualize predictions on test images
plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    image = imread(x_test_paths[i])
    image = resize(image, (128, 128))
    plt.imshow(image, cmap='gray')

    predicted_label = int(y_pred[i])
    true_label = int(y_test_raw[i])
    color = 'green' if predicted_label == true_label else 'red'
    plt.xlabel(f"{class_names[predicted_label]} ({class_names[true_label]})", color=color)

plt.tight_layout()
plt.show()
```

## 🚀 Future Steps

1. **Model Optimization:** Further refine the SVM hyperparameters using more advanced optimization techniques like **RandomizedSearchCV**.
2. **Enhancing Accuracy:** Consider other feature extraction techniques, such as **Haar Cascades** or **Deep Learning** models, to further improve detection accuracy.
3. **Deployment:** Prepare the model for deployment by converting it into a format compatible with web or mobile applications, allowing for real-time pedestrian detection.
4. **Dataset Expansion:** Increase the dataset size by including a more diverse set of images, possibly augmenting data to improve model robustness.
5. **Model Evaluation:** Perform cross-validation and test on real-world datasets to evaluate model performance beyond the current dataset.

---
