import torch 
import torch.nn as nn 
import configs
from tqdm import tqdm 
import time
from dataset import * 
from model import * 
from loss import *
from utils.log_writer import *

def confusion_matrix(predictions : torch.Tensor, labels : torch.Tensor, num_class : int):
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
    conf_matrix = torch.zeros((num_class, num_class), dtype = torch.int64)

    for p, t in zip(predictions.view(-1), labels.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    return conf_matrix

def eval_metrics_bundle(conf_matrix : torch.Tensor, avg_mode = "macro"):
    """
    Calculates precision, recall, and accuracy from a confusion matrix using either macro or micro averaging.

    Args:
        conf_matrix (torch.Tensor): The confusion matrix from which to calculate the metrics.
        avg_mode (str): The averaging mode, 'macro' (default) or 'micro', determining how metrics are computed.

    Returns:
        Tuple[float, float, float]: A tuple containing the precision, recall, and accuracy, each rounded to four decimal places.
    """
    tp = torch.diag(conf_matrix)

    fn = conf_matrix.sum(dim=-2) - tp

    fp = conf_matrix.sum(dim=-1) - tp

    tn = conf_matrix.sum() - (tp + fn + fp)

    if avg_mode == "macro":
        precision = (tp.float() / (tp + fp + 1e-9).float()).mean()
        recall = (tp.float() / (tp + fn + 1e-9).float()).mean()

    elif avg_mode == "micro":
        precision = tp.sum().float() / (tp.sum() + fp.sum() +1e-9).float()
        recall = tp.sum().float() / (tp.sum() + fn.sum() + 1e-9).float()

    accuracy = (tp.sum() + tn.sum()).float() / (tp.sum() + fn.sum() + fp.sum() + tn.sum()).float()

    return round(precision.item(), 4), round(recall.item(), 4), round(accuracy.item(), 4)


def eval_step(model, data, loss_fn): 
    """
    Performs a single evaluation step, including forward pass, loss computation, and metric evaluation.

    Args:
        model (nn.Module): The model to evaluate.
        data (tuple): A tuple containing the inputs and labels from the dataloader.
        loss_fn (callable): The loss function to use for evaluating model performance.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the loss, precision, recall, and accuracy, each rounded to four decimal places.
    """
    image, labels = data
    predictions = model(image)
    loss = loss_fn(predictions, labels)

    precision, recall, accuracy = eval_metrics_bundle(confusion_matrix(predictions=predictions, 
                                                                       labels=labels, 
                                                                       num_class=configs.num_class))
    
    return round(loss.item(), 4), round(precision.item(), 4), round(recall.item(), 4), round(accuracy.item(), 4)

def evaluation(model, logger, loss_fn, dataloader): 
    """
    Evaluates the model over an entire dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        logger (LOGWRITER): Logger instance to record evaluation results.
        loss_fn (callable): The loss function to use.
        dataloader (DataLoader): The DataLoader providing the dataset.

    Side Effects:
        Logs the average metrics for the evaluation to the logger.
    """
    model.eval()

    batched_values = [] 
    for i, data in tqdm(enumerate(dataloader), desc=f"[Evaluation protocol]"): 
        start = time.time() 
        loss, precision, recall, accuracy = eval_step(model, data, loss_fn)
        end = time.time() - start 
        batched_values.append([loss, precision, recall, accuracy, end])
    
    averaged_values = torch.tensor(batched_values).sum(dim=-1) / len(batched_values)
    
    logger.write(epoch=0, CrossEntropyLoss=averaged_values[0], precision=averaged_values[1], recall=averaged_values[2], accuracy=averaged_values[3], runtime=averaged_values[4])

def main(): 
    dataloader = load_dataset(mode="valid")

    model = ViT((3, configs.img_height, configs.img_width), patch_size=8, layers=12, num_classes=configs.num_class).to(configs.device)

    if configs.model_save_path: 
        model.load_state_dict(torch.load(configs.model_save_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))

    logger = LOGWRITER(configs.output_dir, 0)

    loss_fn = CrossEntropyLoss()

    evaluation(model, logger, loss_fn, dataloader)

if __name__ == "__main__": 
    main()