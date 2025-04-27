# predict.py
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from config import Config

def predict_image(model, image_path, idx_to_class):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((Config.resize_x, Config.resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    predicted_class = idx_to_class[pred_idx.item()]

    plt.imshow(np.array(img))
    plt.title(f"Predicted: {predicted_class} ({conf.item()*100:.2f}%)")
    plt.axis('off')
    plt.show()

    return predicted_class, conf.item()
