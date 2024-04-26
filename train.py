import torch 
import torch.nn as nn 
from model import * 
from loss import *
from dataset import * 
from configs import *
from tqdm import tqdm
import time
from utils.log_writer import LOGWRITER

def train_step(model, opt, data, loss_fn):
    """
    """
    opt.zero_grad()

    image, label = data 
    predictions = model(image)
    loss = loss_fn(predictions, label)
    loss.backward()

    opt.step() 
    return loss.item()

def train(model, opt, scheduler, dataloader, logger, loss_fn, epochs): 
    """
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
    dataloader = load_dataset()

    model = ViT((3, configs.img_height, configs.img_width), patch_size=8, layers=12, num_classes=configs.num_class).to(configs.device)

    optimizer = get_optimizer(model, lr = 1e-4, betas=(0.9, 0.999), weight_decay=1e-3)

    scheduler = get_scheduler(optimizer=optimizer, step_size=configs.epochs//5, gamma=0.5)

    logger = LOGWRITER(configs.output_dir, configs.epochs)

    loss_fn = CrossEntropyLoss()

    train(model, optimizer, scheduler, dataloader, logger, loss_fn)

if __name__ == "__main__": 
    main()