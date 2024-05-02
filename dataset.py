import torch 
from PIL import Image
import os
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from glob import glob
import configs

class ImgClsDataset(Dataset):
    def __init__(self, root_dir, img_height, img_width, mode, transforms = None):
        
        self.data_dir = os.path.join(root_dir, mode)
        
        self.img_height = img_height 
        self.img_width = img_width 

        if transforms: 
            self.transforms = transforms 
        else: 
            self.transforms = T.Compose([T.Resize((self.img_height, self.img_width), T.InterpolationMode.BICUBIC), 
                                         T.RandomHorizontalFlip(0.25), 
                                         T.RandomVerticalFlip(0.25), 
                                         T.GaussianBlur((5,5), sigma=(0.1, 2)),
                                         T.ToTensor()])
        self.id_to_class_dict = {} 
        self.class_to_id_dict = {}
        self.images = []

        for i, cls in enumerate(os.listdir(self.data_dir)): 
            self.id_to_class_dict[i] = cls 
            self.class_to_id_dict[cls] = i

            if os.path.isdir(os.path.join(self.data_dir, cls)):
                for image in glob(os.path.join(self.data_dir, cls) + "/*"):
                    self.images.append((image, i))

    def __getitem__(self, index):
        img_path, img_id = self.images[index]
        img = self.transforms(Image.open(img_path).convert("RGB"))
        return img.to(configs.device), img_id

    def __len__(self):
        return len(self.images)
    
def load_dataset(img_height = configs.img_height, img_width = configs.img_width, batch_size = configs.batch_size, shuffle = True, mode = "train"): 
    assert mode in ["train", "val", "test"], "[ERROR] Invalid dataset mode"
    ds = ImgClsDataset(configs.root_dir, img_height, img_width, mode = mode, transforms = configs.transforms)
    configs.num_class = len(ds.id_to_class_dict)
    print(f"[INFO] Total Class count: {configs.num_class}")
    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)