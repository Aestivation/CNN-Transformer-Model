# CNN-Transformer KITTI Object Detection

This project implements an object detection model using a hybrid **CNN + Transformer** architecture, trained on the **KITTI dataset**. The model predicts bounding boxes for objects in images, utilizing convolutional layers for feature extraction and a transformer block for spatial attention.

## Dataset and Preprocessing  
- The **KITTI dataset** was used, containing labeled images for object detection.  
- A subset of **1000 samples** was selected for training to balance computational efficiency and performance.  
- Images were resized to **224x224 pixels**, and bounding box coordinates were normalized between **0 and 1**.  
- **Data Augmentation** was applied, including random rotation, horizontal flipping, and width/height shifts.

## Model Architecture  
The model consists of two main components:  

### 1. CNN Backbone  
   - A **VGG16** model (pre-trained on ImageNet) is used as the feature extractor.  
   - Global Average Pooling (GAP) reduces spatial dimensions.  

### 2. Transformer Block  
   - Multi-head self-attention captures dependencies between features.  
   - A feed-forward network (FFN) refines the learned representations.  

### 3. Bounding Box Prediction  
   - The output layer consists of **4 neurons (x1, y1, x2, y2)**, each activated with **sigmoid** to keep values between 0 and 1.

## Training Details  
- The model was trained for **20 epochs** with a batch size of **32**.  
- **Mean Squared Error (MSE)** was used as the loss function.  
- The optimizer was **Adam** with a learning rate of **1e-4**.  
- **Early stopping** was implemented to prevent overfitting, stopping training if validation loss did not improve for **5 consecutive epochs**.  
- A **model checkpoint** was used to save the best performing model.

## Evaluation and Testing  
- The last **20 training samples** were used for testing.  
- The model’s predictions were compared against ground truth bounding boxes.  
- The results were visualized by plotting bounding boxes on the test images.

## How to Continue Analysis  
- If deeper analysis is required, users can:  
  - Increase the dataset size beyond **1000 samples** to improve generalization.  
  - Fine-tune the learning rate and batch size for better convergence.  
  - Modify the Transformer block’s hyperparameters (number of heads, embedding dimension).  
  - Experiment with different CNN architectures, such as **ResNet50** or **EfficientNet**.

## Usage Instructions  
1. Mount Google Drive and download the KITTI dataset.  
2. Extract the dataset and ensure paths are correctly set.  
3. Run the training script and monitor loss values.  
4. Save and reload the trained model for inference.  
5. Evaluate performance using the test set and visualize bounding boxes.  

For improvements, adjusting hyperparameters and testing with different backbone networks can help refine results.
