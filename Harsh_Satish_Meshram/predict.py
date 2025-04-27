import torch
from torchvision import transforms
from PIL import Image
import os
from config import Config

def predict_batch(model, data_dir, idx_to_class):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((Config.resize_x, Config.resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predictions = []

    # Iterate through directories in data_dir (each directory is an exercise category)
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)

                # Open and preprocess the image
                img = Image.open(image_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)

                # Make prediction
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)

                predicted_class = idx_to_class[pred_idx.item()]
                predictions.append((image_path, predicted_class, conf.item()))

    return predictions
