import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = "/workspace/cifar10/cifar10"

img_height = 32
img_width = 32

transforms = None

model_save_path = None

batch_size = 128

epochs = 35

output_dir = "/workspace"

num_class = None