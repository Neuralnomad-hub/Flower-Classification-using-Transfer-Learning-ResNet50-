# 🌸 Flower Classification using Transfer Learning (ResNet50)

## 1. **Problem Statement**

This project builds a **deep learning image classification model** to identify five categories of flowers — *daisy*, *dandelion*, *roses*, *sunflowers*, and *tulips*.
By leveraging **transfer learning with ResNet50**, the model achieves **high-accuracy predictive insights** that can support **botanical research, garden automation, and plant monitoring applications**.

---

## 2. **Dataset Overview**

* **Source**: [TensorFlow Flower Photos Dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
* **Size**: 3,670 images, 5 classes
* **Classes**:

  1. Daisy
  2. Dandelion
  3. Roses
  4. Sunflowers
  5. Tulips
* **Image Dimensions**: Resized to **180×180 px** for uniformity
* **Data Split**:

  * Training: 2,936 images (80%)
  * Validation: 734 images (20%)
* **Type**: RGB color images

---

## 3. **Data Collection & Cleaning**

* **Automated download** from the TensorFlow public dataset repository
* **Preprocessing steps**:

  * Image resizing to `(180, 180, 3)`
  * Normalization to scale pixel values
  * Dataset splitting into training and validation sets
* **Data integrity check** to ensure correct class labeling

---

## 4. **Exploratory Data Analysis (EDA)**

* **Class distribution**: Nearly balanced across the 5 categories
* **Sample visualization**: Grid display of random training images per class
* Observed **intra-class variation** (e.g., roses with different colors and lighting conditions)
* **Business insight**: Model needs to be robust to background noise and color differences

---

## 5. **Machine Learning Model**

* **Architecture**:

  * Pre-trained **ResNet50** (ImageNet weights, top layers removed)
  * Added layers:

    * `GlobalAveragePooling2D`
    * Dense layer with 512 ReLU units
    * Dense output layer with Softmax activation for 5-class classification
* **Transfer Learning**: Base ResNet50 layers frozen during training
* **Optimizer**: Adam
* **Loss Function**: Sparse Categorical Crossentropy
* **Batch Size**: 32
* **Epochs**: 10

---

## 6. **Model Evaluation**

| Metric       | Training Accuracy | Validation Accuracy |
| ------------ | ----------------- | ------------------- |
| **Accuracy** | 100%              | 88.69%              |
| **Loss**     | 0.0028            | 0.3923              |

**Insights**:

* The model converges quickly due to pre-trained features.
* Slight overfitting observed after 5th epoch — can be addressed via fine-tuning or regularization.

---

## 7. **Results & Business Impact**

* Successfully classifies flowers with **high accuracy**.
* Potential **real-world applications**:

  * Automated **botanical species identification**
  * Integration into **gardening mobile apps**
  * Use in **environmental monitoring systems**
* **ATS keywords**: image recognition, computer vision, deep learning, transfer learning, ResNet50, TensorFlow

---

## 8. **Future Improvements**

* Fine-tune top ResNet50 layers to improve validation accuracy
* Apply data augmentation to reduce overfitting
* Deploy as a **TensorFlow Lite model** for mobile applications
* Integrate into a **Streamlit/Flask web app** for live image classification

---

## 9. **Tech Stack**

* **Programming Language**: Python
* **Libraries**: TensorFlow, Keras, NumPy, Matplotlib, OpenCV, PIL
* **Model**: ResNet50 (Transfer Learning)
* **Hardware**: GPU-enabled training environment
---

## 11. **Sample Prediction**

**Input Image:** 🌹 Rose
**Predicted Output:** `roses` (confidence: 99.82%)

---

## 12. **Author**

👤 **Mohammed Raiyan**
*Aspiring Data Scientist | Machine Learning Enthusiast | AI Solutions Builder*
