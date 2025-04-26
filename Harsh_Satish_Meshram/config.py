resize_x, resize_y = 128, 128
input_channels = 3
batchsize = 32
epochs = 15
learning_rate = 0.001
num_classes = 22
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = '/kaggle/working/workoutexercises-images/train'
test_dir = '/kaggle/working/workoutexercises-images/test'
