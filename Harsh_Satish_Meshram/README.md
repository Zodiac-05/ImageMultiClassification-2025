# Workout Exercise Image Classification

## Project Description

This project performs **multiclass classification** of **workout exercise images**.  
Given an image of a person performing an exercise, the model predicts the correct exercise name.

The project was done as part of an assignment for image classification using deep learning.

---

## Dataset

- The dataset used is:  
  📦 [Workout Exercises Dataset (Images Only) by Harvineet Singh](https://www.kaggle.com/datasets/hasyimabdillah/workoutexercises-images)
- Size: ~800 MB
- It contains images extracted from videos and Google images.
- Classes include exercises like `bench press`, `barbell biceps curl`, `plank`, `deadlift`, etc.

**Note**:  
The dataset is NOT uploaded here due to size limits.  
Please download the dataset manually from Kaggle and extract it under `data/workout-images/`.

---

## Model Choices

Two models were used:

1. **EfficientNetB0** (Pretrained on ImageNet):
   - Used transfer learning by fine-tuning last few layers.
   - Achieved very high accuracy (~90% validation accuracy).

2. **Custom CNN** (built from scratch):
   - 4 convolutional layers followed by fully connected layers.
   - Provides more control and no dependency on pretrained weights.

**Switching Models**:  
You can easily switch between models by changing one line in `config.py`:

```python
model_name = 'EfficientNetB0'  # for pretrained EfficientNet
# OR
model_name = 'CustomCNN'  # for training custom CNN from scratch

---

## Data Leakage
One of the significant challenges in this project is that the images were extracted from video frames. Similar frames from the same video may end up in both the training and test datasets, leading to data leakage. This can significantly inflate the model's performance and create an unrealistic estimation of its true capability when generalised to unseen data.
To address this, I implemented a group-based split where each group corresponds to frames from a particular video. This ensures that no image from the same video appears in both the training and test sets, preventing any potential leakage from similar video frames.

---
