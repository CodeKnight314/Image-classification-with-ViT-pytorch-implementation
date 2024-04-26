import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = ""

img_height = ""
img_width = ""
data_transform = ""

model_save_path = ""

batch_size = 32

epochs = 20 

output_dir = ""

num_class = None