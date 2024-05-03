import torch 
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = "/workspace/food-dataset"

img_height = 128
img_width = 128

transforms = None

model_save_path = None

batch_size = 4

epochs = 50

output_dir = "/workspace"

log_output_dir = os.path.join(output_dir, "log_outputs")

matrix_output_dir = os.path.join(output_dir, "conf_matrices")

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

if __name__ == "__main__": 
    main()
    