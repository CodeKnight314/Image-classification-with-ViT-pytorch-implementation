import torch 
import torch.nn as nn 
from loss import *
from dataset import * 
from configs import *
from tqdm import tqdm
import time
from utils.log_writer import LOGWRITER
from ViT import * 
from ResNet import *

def train_step(model, opt, data, loss_fn):
    """
    Executes a single training step, which includes forward passing the data through the model, calculating the loss,
    performing backpropagation, and updating the model's weights.

    Args:
        model (torch.nn.Module): The model being trained.
        opt (torch.optim.Optimizer): Optimizer used to update the model parameters.
        data (tuple): A tuple containing the input data and labels. Expects (image, label).
        loss_fn (callable): The loss function used to evaluate the model's predictions against the true labels.

    Returns:
        float: The loss value calculated for this training step.
    """
    opt.zero_grad()

    image, label = data 
    label = label.to(configs.device)
    predictions = model(image)
    loss = loss_fn(predictions, label)
    loss.backward()
    opt.step() 
    return loss.item()

def train(model, opt, scheduler, dataloader, logger, loss_fn, epochs): 
    """
    Manages the training process over multiple epochs. It logs training progress, updates the learning rate, 
    and saves the model checkpoints when a new best is achieved based on validation loss.

    Args:
        model (torch.nn.Module): The model to be trained.
        opt (torch.optim.Optimizer): Optimizer used to update the model's weights.
        scheduler (torch.optim.lr_scheduler): Scheduler to adjust the optimizer's learning rate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training data.
        logger (LOGWRITER): Logger instance used to log training metrics.
        loss_fn (callable): Loss function used to compute the model's prediction error.
        epochs (int): Total number of epochs to train the model.
    """
    model.train() 

    best_loss = float('inf')

    for epoch in range(epochs):

        batched_values = []
        for i, data in tqdm(enumerate(dataloader), desc=f"[Training: {epoch+1}/{epochs}]"): 
            start = time.time()
            loss = train_step(model, opt, data, loss_fn)
            end = time.time() - start
            batched_values.append([loss, end])

        averaged_values = torch.tensor(batched_values).sum(dim=-1) / len(batched_values)

        logger.write(epoch=epoch, CrossEntropyLoss=averaged_values[0], runtime = averaged_values[1])

        if best_loss > averaged_values[0]: 
            if not os.path.exists(configs.model_save_path):
                os.makedirs(configs.model_save_path) 
            torch.save(model.state_dict(), os.path.join(configs.model_save_path, f"FRCNN_model_{epoch+1}.pth"))
            best_loss = averaged_values[0]

        scheduler.step()

def main(): 
    dataloader = load_dataset(mode="train")

    model = ViT((3, configs.img_height, configs.img_width), patch_size=configs.ViT_patches, layers=configs.ViT_layers, num_classes=configs.num_class).to(configs.device)

    optimizer = get_optimizer(model, lr = 1e-4, betas=(0.9, 0.999), weight_decay=1e-3)

    scheduler = get_scheduler(optimizer=optimizer, step_size=configs.epochs//5, gamma=0.5)

    logger = LOGWRITER(configs.log_output_dir, configs.epochs)

    loss_fn = CrossEntropyLoss()

    train(model, optimizer, scheduler, dataloader, logger, loss_fn)

if __name__ == "__main__": 
    main()