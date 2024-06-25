import cv2 
import numpy as np
from typing import Tuple, Union
from glob import glob 
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import configs
from collections import Counter
import math

def add_gaussian_noise(image_path : str, 
                       mean : int, 
                       std : int, 
                       output_directory : Union[str, None], 
                       show : bool = False) -> None: 
    """
    Adds Gaussian Noise to a given image with a specified mean and standardeviation 

    Args: 
        image_path (str): direct directory to the image 
        mean (int): mean distribution of gaussian noise 
        std (int): standard deviation of the guassian noise distribution
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key
    """ 
    image = cv2.imread(image_path)

    gaussian_noise = np.zeros(image.shape, dtype = np.uint8)
    
    cv2.randn(gaussian_noise, mean=mean, std=std)
    
    gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
    
    image = cv2.add(image, gaussian_noise)
    
    if show: 
            cv2.imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,image)

    return image

def add_uniform_noise(image_path : str, 
                      output_directory : Union[str, None], 
                      lower_bound : int, 
                      upper_bound : int, 
                      show : bool = False) -> None:
    """
    Adds Uniform Noise to a given image with a specified lower and upper bound. 

    Args: 
        image_path (str): direct directory to the image 
        lower_bound (int): lower bound of the uniform distribution
        upper_bound (int): upper bound of the uniform distribution
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key
    """
    image = cv2.imread(image_path)

    uni_noise = np.zeros(image.shape, dtype = np.unint8)

    cv2.randu(uni_noise, low=lower_bound, high=upper_bound)

    image = cv2.add(image, uni_noise)

    if show: 
            cv2.imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,image)

    return image

def add_impulse_noise(image_path : str, 
                      output_directory : Union[str, None], 
                      lower_bound : int, 
                      upper_bound : int, 
                      show : bool = False) -> None:
    """
    Adds Impulse Noise (Pepper Noise) to a given image.
    
    Args: 
        image_path (str): direct directory to the image 
        lower_bound (int): lower bound of the uniform distribution
        upper_bound (int): upper bound of the uniform distribution
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key

    """
    image = cv2.imread(image_path)

    imp_noise = np.zeros(image.shape, dtype = np.unint8)

    cv2.randu(imp_noise, low=lower_bound, high=upper_bound)
    
    imp_noise = cv2.threshold(imp_noise,245,255,cv2.THRESH_BINARY)[1]

    image = cv2.add(image, imp_noise)

    if show: 
            cv2.imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,image)

    return image

def batch_noise(root_dir : str, 
                output_dir : Union[str, None], 
                mode : str = "gaussian", **kwargs) -> None:
    """
    Applies noise to all images in a specified directory and saves the modified images to an output directory. 
    The function supports different types of noise such as Gaussian, uniform, and impulse.

    Args:
        root_dir (str): The directory containing the images to process.
        output_dir (Union[str, None]): The directory where the noised images will be saved. If None, images are not saved.
        mode (str): The type of noise to apply. Options include 'gaussian', 'uniform', or 'impulse'.
        **kwargs: Keyword arguments specific to the type of noise:
            For 'gaussian':
                mean (float): The mean of the Gaussian noise.
                std (float): The standard deviation of the Gaussian noise.
            For 'uniform' and 'impulse':
                lower_bound (float): The lower bound of the noise distribution.
                upper_bound (float): The upper bound of the noise distribution.

    Raises:
        ValueError: If an invalid mode is specified.
    """
    image_paths = glob(os.path.join(root_dir, "/*"))
    if mode.lower() == "gaussian":           
        for image in tqdm(image_paths): 
            add_gaussian_noise(image_path=image, 
                               output_directory=os.path.join(output_dir, os.path.basename(image).split("/")[-1]), 
                               mean = kwargs['mean'], std=kwargs['std'], 
                               show=False)
    elif mode.lower() == "uniform": 
        for image in tqdm(image_paths): 
            add_uniform_noise(image_path=image, 
                              output_directory=os.path.join(output_dir, os.path.basename(image).split("/")[-1]),
                              lower_bound=kwargs['lower_bound'], upper_bound=kwargs['upper_bound'], 
                              show=False)
    elif mode.lower() == "impulse":
        for image in tqdm(image_paths): 
            add_uniform_noise(image_path=image, 
                              output_directory=os.path.join(output_dir, os.path.basename(image).split("/")[-1]),
                              lower_bound=kwargs['lower_bound'], upper_bound=kwargs['upper_bound'], 
                              show=False)
    else: 
        raise ValueError(f"[Error] Invalid mode. {mode} is not available as a noise mode.")

def confusion_matrix(predictions: torch.Tensor, 
                     labels: torch.Tensor, 
                     num_class: int) -> torch.Tensor:
    """
    Computes the confusion matrix from predictions and labels.

    Args:
        predictions (torch.Tensor): The tensor containing the predicted class indices for each example.
        labels (torch.Tensor): The tensor containing the actual class indices for each example.
        num_class (int): The total number of classes.

    Returns:
        torch.Tensor: A confusion matrix of shape (num_class, num_class) where the element at position (i, j)
                      represents the number of instances of class i predicted as class j.
    """
    conf_matrix = torch.zeros((num_class, num_class), dtype=torch.int64)

    for p, t in zip(predictions.view(-1), labels.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    return conf_matrix 

def plot_confusion_matrix(confusion_matrix : torch.Tensor, 
                          num_classes : int, 
                          save_pth : Union[str, None]) -> None:
    """
    Computes and plots a confusion matrix.
    
    Args:
        true_labels (np.array): 1D NumPy array of true class labels
        predictions (np.array): 1D NumPy array of predicted class labels
        num_classes (np.array): Total number of classes
        save_pth (Union[str, None]): save path for confusion matrix 
    """
    
    cm = confusion_matrix.detach().cpu().numpy()
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot = True, fmt='d', cmap='Blues', 
                xticklabels=[f"{configs.id_to_category_dict[i]}" for i in range(num_classes)], 
                yticklabels=[f"{configs.id_to_category_dict[i]}" for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_pth: 
        plt.savefig(save_pth)

    plt.close()

def count_labels(directory : str) -> dict:
    """
    Counts the number of images in each label directory within the given parent directory.
    
    Args:
    directory (str): The path to the directory containing labeled subdirectories of images.
    
    Returns:
    dict: A dictionary with keys as labels and values as counts of images.
    """
    label_counts = Counter()
    # Iterate through each subdirectory in the given directory
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            # Count files in the subdirectory, assuming all are images
            label_counts[label] = len([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
    
    return dict(label_counts)

def plot_data(output_log : str, 
              output_directory : str, 
              save_fig : bool, 
              show_fig : bool) -> None: 
    """"
    Plots training and validation loss as well as accuracy over epochs from a log file.

    Parameters:
    output_log (str): Path to the log file containing training and validation metrics.
    output_directory (str): Directory where the plots will be saved if save_fig is True.
    save_fig (bool): If True, the plots will be saved to the specified directory.
    show_fig (bool): If True, the plots will be displayed.
    """
    file = open(output_log, 'r').split("\n")

    tr_loss_ls = []
    val_loss_ls = []
    accuracy_ls = []
     
    for i in range(len(file)):
        line = file[i].split(" ")

        tr_loss_ls.append(line[4])
        val_loss_ls.append(line[8])
        accuracy_ls.append(line[12])
    
    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(1, len(tr_loss_ls) + 1)], tr_loss_ls, label="Training Loss")
    plt.plot([i for i in range(1, len(val_loss_ls) + 1)], val_loss_ls, label="Validation loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend() 

    if save_fig: 
        plt.savefig(os.path.join(output_directory, "Training_Validation_Loss.png"))

    if show_fig: 
        plt.show() 
    
    plt.close() 

    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(1, len(accuracy_ls) + 1)], accuracy_ls, label="Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend() 

    if save_fig: 
        plt.savefig(os.path.join(output_directory, "Accuracy.png"))
    
    if show_fig: 
        plt.show() 

    plt.close()

def heatmap_generation(input_tensor : torch.Tensor, attn_weights : torch.Tensor, patch_size : int, output_filename : Union[str, None], save_fig : bool, show_fig : bool): 
    """
    Generates the heatmap for a given input tensor based on the attention weights provided. 

    Args: 
        input_tensor (torch.Tensor): Image tensor to generate heatmap for with dimensiosn (C, H, W).
        att_weights (torch.Tensor): Attention weights provided to generate heatmap with dimensions (num_of_heads, num_patches, num_patches)
        patch_size (int): Patch size used for classification inference.
        output_filename (Union[str, None]): Output filename that includes the directory address and filename to save figure to.
        save_fig (bool): Flag to enable saving figure to specified the designated address and filename
        show_fig (bool): Flag to enable showing figure for given image and attention weights.
    """
    attn_weights = attn_weights.mean(dim=0).cpu().detach().numpy()
    num_patches = attn_weights.shape[-1]

    img = input_tensor.cpu().detach().numpy()
    img_h, img_w = img.shape[-1], img.shape[-2]

    patch_h, patch_w = img_h // patch_size, img_w // patch_size

    fig, ax = plt.subplots()
    heatmap = np.zeros((img_h, img_w))
    for i in range(num_patches):
        row_index = i // patch_w
        col_index = i // patch_w
        patch = attn_weights[:, i].reshape((patch_size, patch_size))
        heatmap[row_index * patch_size:(row_index+1)*patch_size, col_index*patch_size:(col_index+1)*patch_size] = patch

    ax.imshow(img.permute(1, 2, 0).cpu().detach().numpy())
    ax.imshow(heatmap, cmap='viridis', alpha=0.6)

    if save_fig:
        fig.savefig(output_filename)

    if show_fig:
        plt.show()

    plt.close()
    

