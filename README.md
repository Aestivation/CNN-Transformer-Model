# CNN + Transformer Hybrid Model for KITTI Dataset

This project implements an object detection model using **CNN (Convolutional Neural Networks) and Transformers** to detect objects in images from the **KITTI dataset**, which is widely used in autonomous driving research.

---

##  Project Overview
The goal of this project is to **accurately detect and localize objects** (such as **cars, pedestrians, and obstacles**) in images from the **KITTI dataset** using a **hybrid deep learning approach** that combines:
✔ **CNN (ResNet50)** for **feature extraction**  
✔ **Transformer Encoder** for **context-aware object detection**  

---

## Dataset: KITTI
We use the **KITTI Object Detection dataset**, which consists of **stereo images captured from a moving vehicle**.  
The dataset contains:
- **Images:** RGB road scenes (`224×224×3`)
- **Labels:** Bounding boxes (`x1, y1, x2, y2`)

 **Dataset Source**: [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)

---

##  Model Architecture
1️⃣ **CNN Backbone (ResNet50)**
   - Extracts low- and high-level features from images  
   - Pre-trained on ImageNet  
   - Output shape: `(49, 2048)`  

2️⃣ **Transformer Encoder**
   - Uses **multi-head self-attention** to capture spatial relationships  
   - Fully connected layers process features  
   - Output shape: `(49, 512)`  

3️⃣ **Bounding Box Prediction**
   - Dense layers predict `(x1, y1, x2, y2)` coordinates  
   - Output normalized between `0-1` using **Sigmoid activation**  

---

##  Installation
To run this project, install the required libraries:
\`\`\`bash
pip install tensorflow numpy matplotlib opencv-python kaggle
\`\`\`

---

##  How to Run
1️⃣ **Download the KITTI dataset from Kaggle**  
   - Set up your **Kaggle API key** (`kaggle.json`)  
   - Run the following in Colab:
   \`\`\`bash
   kaggle datasets download -d klemensko/kitti-dataset
   unzip -q kitti-dataset.zip -d /content/kitti_data/
   \`\`\`

2️⃣ **Train the model**
   \`\`\`python
   model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
   \`\`\`

3️⃣ **Test on new images**
   \`\`\`python
   predicted_boxes = model.predict(sample_images)
   \`\`\`

4️⃣ **Visualize results**
   \`\`\`python
   plt.imshow(image_with_bounding_box)
   \`\`\`

---

## Results & Performance
- The model successfully detects objects in KITTI images.
- The Transformer helps improve detection accuracy in complex scenes.
- Further fine-tuning and augmentation can enhance performance.

---

##  Future Improvements
- **Use a more advanced CNN (EfficientNet) for better feature extraction.**  
- **Apply fine-tuning on Transformer layers for better contextual learning.**  
- **Experiment with additional post-processing techniques (e.g., Non-Maximum Suppression - NMS).**  

---



