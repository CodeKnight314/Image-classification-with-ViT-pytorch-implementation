import torch 
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = "/workspace/cifar10/cifar10"

img_height = 32
img_width = 32

transforms = None

model_save_path = None

batch_size = 128

epochs = 200

patches = 2

output_dir = "/workspace"

log_output_dir = os.path.join(output_dir, "log_outputs")

matrix_output_dir = os.path.join(output_dir, "conf_matrices")

save_pth = os.path.join(output_dir, "saved_weights")

heatmaps = os.path.join(output_dir, "heatmaps")

num_class = None

id_to_category_dict = None

category_to_id_dict = None

def main(): 
    if not os.path.exists(log_output_dir): 
        os.makedirs(log_output_dir)
        print("[INFO] Log output dir not found, Creating folder directory.")

    if not os.path.exists(matrix_output_dir): 
        os.makedirs(matrix_output_dir)
        print("[INFO] Matrix output dir not found. Creating folder directory.")

    if not os.path.exists(save_pth): 
        os.makedirs(save_pth)
        print("[INFO] Model Save path not found. Creating folder directory.")
    
    if not os.path.exists(heatmaps):
        os.makedirs(heatmaps)
        print("[INFO] Heat map directory not found. Creating folder directory.")

if __name__ == "__main__": 
    main()
    