import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = "/workspace/food-101-dataset"

img_height = 256
img_width = 256

transforms = None

model_save_path = None

batch_size = 32

epochs = 20 

output_dir = "/workspace"

num_class = None