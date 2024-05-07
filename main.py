from train import * 
from eval import * 
from model import * 
from dataset import * 
from loss import * 
from utils.log_writer import * 
import configs 

torch.autograd.set_detect_anomaly(True)

def train_and_evaluation(model : ViT,
                         optimizer : torch.optim, 
                         scheduler : torch.optim.lr_scheduler, 
                         train_dl : DataLoader,
                         valid_dl : DataLoader, 
                         logger : LOGWRITER, 
                         loss_fn : CrossEntropyLoss, 
                         epochs : int): 
    """
    Conducts training and validation of a Vision Transformer model over a specified number of epochs, logs the 
    performance metrics, and saves the best model based on validation loss.

    Args:
        model (ViT): The Vision Transformer model to be trained and evaluated.
        optimizer (torch.optim): The optimizer to use for training the model.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler to adjust the learning rate during training.
        train_dl (DataLoader): The DataLoader for the training data.
        valid_dl (DataLoader): The DataLoader for the validation data.
        logger (LOGWRITER): An instance of LOGWRITER for logging training and validation metrics.
        loss_fn (CrossEntropyLoss): The loss function used for training the model.
        epochs (int): The total number of epochs to train the model.
    """
    
    best_loss = float('inf')
    
    for epoch in range(epochs): 

        train_batched_values = [] 
        for data in tqdm(train_dl, desc=f"[Training: {epoch+1}/{epochs}]"): 
            loss = train_step(model=model, opt=optimizer, data=data, loss_fn=loss_fn)

            train_batched_values.append([loss])
        
        valid_batched_values = [] 
        pred_stack = torch.tensor([]).to(configs.device)
        label_stack = torch.tensor([]).to(configs.device)

        for data in tqdm(valid_dl, desc = f"[Validating {epoch+1}/{epochs}]"):
            loss, precision, recall, accuracy, predictions, labels = eval_step(model=model, data=data, loss_fn=loss_fn)
            pred_stack = torch.cat([pred_stack, predictions])
            label_stack = torch.cat([label_stack, labels])
            valid_batched_values.append((loss, precision, recall, accuracy))
        
        avg_train_loss = torch.tensor(train_batched_values).sum(dim=1) / len(train_batched_values)
        avg_valid_value = torch.tensor(valid_batched_values).sum(dim=1) / len(valid_batched_values)

        conf_matrix = confusion_matrix(predictions=pred_stack, labels=label_stack, num_class=configs.num_class)
        plot_confusion_matrix(confusion_matrix=conf_matrix, num_classes=configs.num_class, save_pth=os.path.join(configs.matrix_output_dir, f"Epoch_{epoch+1}_conf_matrix.png"))

        if best_loss > avg_train_loss[0].item(): 
            if not os.path.exists(os.path.join(configs.output_dir, "saved_weights")):
                os.makedirs(os.path.join(configs.output_dir, "saved_weights"))
                
            torch.save(model.state_dict, os.path.join(os.path.join(configs.output_dir, "saved_weights"), f"ResNet_{configs.img_height}x{configs.img_width}_{epoch+1}.pth"))
            best_loss = avg_train_loss[0].item()

        logger.write(epoch=epoch+1, 
                     tr_loss = avg_train_loss[0].item(), 
                     avg_valid_loss = avg_valid_value[0].item(), 
                     precision = avg_valid_value[1].item(), 
                     recall = avg_valid_value[2].item(), 
                     accuracy = avg_valid_value[3].item())
        
        scheduler.step()

def main():
    configs.main()

    train_dl = load_dataset(mode = "train")
    valid_dl = load_dataset(mode = "test")

    loss_fn = nn.CrossEntropyLoss() #Custom implementation is kinda screwed rn. 

    model = get_ResNet(num_classes = configs.num_class)
    if configs.model_save_path: 
        print("[INFO] Model weights provided. Loading model weights to ViT.")
        model.load_state_dict(torch.load(configs.model_save_path))
    
    optimizer = get_optimizer(model=model, lr = 1e-4, betas=(0.9,0.999), weight_decay=1e-5)

    scheduler = get_scheduler(optimizer=optimizer, step_size = configs.epochs, gamma=0.5)

    logger = LOGWRITER(output_directory=configs.log_output_dir, total_epochs=configs.epochs)

    train_and_evaluation(model=model, optimizer=optimizer, scheduler=scheduler, train_dl=train_dl, valid_dl=valid_dl, logger=logger, loss_fn=loss_fn, epochs=configs.epochs)

if __name__ == "__main__": 
    main()