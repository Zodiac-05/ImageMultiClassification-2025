import torch
from torchvision import transforms
from PIL import Image
import os
from config import Config

def predict_batch(images=None):
    """
    Predict the class label for a batch of images in the specified directory.
    
    Args:
    - images (list): A list of image paths to predict, if provided. 
      If no images are provided, it will use `Config.data_dir` for predictions.

    Returns:
    - List of tuples: Each tuple contains (true_class, predicted_class)
    """

    # Set the device to GPU if available, otherwise fall back to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set the model to evaluation mode
    
    idx_to_class = {idx: class_name for idx, class_name in enumerate(os.listdir(Config.data_dir))}

    # Preprocessing steps: Resize, Convert to Tensor, and Normalize the image
    preprocess = transforms.Compose([
        transforms.Resize((Config.resize_x, Config.resize_y)),  # Resize the image
        transforms.ToTensor(),  # Convert image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])

    predictions = []  # List to store the predictions

    if images:
        for image_path in images:
            img = Image.open(image_path).convert('RGB')  # Convert to RGB if not already
            img_tensor = preprocess(img).unsqueeze(0).to(device)  # Preprocess and add batch dimension

            # Make prediction using the trained model
            with torch.no_grad():  # Disable gradient calculation to save memory
                outputs = model(img_tensor)  # Get model outputs
                probs = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                _, pred_idx = torch.max(probs, 1)  # Get the predicted class index

            # Get the predicted class name from idx_to_class
            idx_to_class = {idx: class_name for idx, class_name in enumerate(os.listdir(Config.data_dir))}
            predicted_class = idx_to_class[pred_idx.item()]

            # Append the prediction (true_class, predicted_class) to the list
            predictions.append((os.path.basename(image_path), predicted_class))
    else:
        # Dynamically create idx_to_class from subdirectory names in the data_dir
        idx_to_class = {idx: class_name for idx, class_name in enumerate(os.listdir(Config.data_dir))}
        # Loop through each subdirectory in the data directory (each subdir is a different exercise)
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            
            # Check if the path is a directory
            if os.path.isdir(class_path):
                
                # Loop through each image in the class subdirectory
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
    
                    # Open and preprocess the image
                    img = Image.open(image_path).convert('RGB')  # Convert to RGB if not already
                    img_tensor = preprocess(img).unsqueeze(0).to(device)  # Preprocess and add batch dimension
    
                    # Make prediction using the trained model
                    with torch.no_grad():  # Disable gradient calculation to save memory
                        outputs = model(img_tensor)  # Get model outputs
                        probs = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                        _, pred_idx = torch.max(probs, 1)  # Get the predicted class index
    
                    # Get the predicted class name from idx_to_class
                    predicted_class = idx_to_class[pred_idx.item()]
    
                    # Append the prediction (image_path, predicted_class) to the list
                    predictions.append((image_path, predicted_class))

    # Return the list of predictions
    return predictions

def load_model():
    """
    Load the pretrained model with weights saved in checkpoints/best_model.pth.

    Returns:
    - model (torch.nn.Module): The model with loaded weights.
    """
    model = EfficientNetModel(num_classes=Config.num_classes)  # Initialize the model architecture
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))  # Load the saved weights
    return model
