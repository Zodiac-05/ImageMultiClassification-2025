# config.py
class Config:
    resize_x = 224
    resize_y = 224
    batchsize = 32
    epochs = 50
    learning_rate = 0.0001
    patience = 5
    test_size = 0.2
    random_seed = 42
    data_dir = 'Harsh_Satish_Meshram/data'
    model_name = 'EfficientNetB0'  # 'EfficientNetB0' or 'CustomCNN'
    save_path = 'checkpoints/best_model.pth'
