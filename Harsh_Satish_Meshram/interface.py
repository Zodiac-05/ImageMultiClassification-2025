# interface.py
from model import EfficientNetModel as TheModel   # If using CustomCNN, change it here
from train import train_model as the_trainer
from predict import predict_image as the_predictor
from dataset import WorkoutDataset as TheDataset
from dataset import create_dataloader as the_dataloader
