import cv2 
import numpy as np
from typing import Tuple, Union
from glob import glob 
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import matplotlib.patches as patches
import configs
from collections import Counter


def add_gaussian_noise(image_path : str, mean : int, std : int, output_directory : Union[str, None], show : bool = False): 
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

def add_uniform_noise(image_path : str, output_directory : Union[str, None], lower_bound : int, upper_bound : int, show : bool = False):
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

def add_impulse_noise(image_path : str, output_directory : Union[str, None], lower_bound : int, upper_bound : int, show : bool = False):
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

def batch_noise(root_dir : str, output_dir : Union[str, None], show : bool = False, mode : str = "gaussian", **kwargs):
    """
    Applies noise to all images in a specified directory and saves the modified images to an output directory. 
    The function supports different types of noise such as Gaussian, uniform, and impulse.

    Args:
        root_dir (str): The directory containing the images to process.
        output_dir (Union[str, None]): The directory where the noised images will be saved. If None, images are not saved.
        show (bool): If True, displays the noised image. Defaults to False.
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

def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, num_class: int):
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

def plot_confusion_matrix(confusion_matrix : torch.Tensor, num_classes : int, save_pth : Union[str, None]):
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

def count_labels(directory):
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