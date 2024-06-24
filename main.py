from models.ViT import *
from models.ResNet import *
from dataset import * 
from loss import * 
from utils.visualization import plot_confusion_matrix, confusion_matrix
from utils.log_writer import * 
import configs 
from torch.nn import functional as F

torch.autograd.set_detect_anomaly(True)

def train_and_evaluate(model, 
                       optimizer : torch.optim, 
                       scheduler : torch.optim.lr_scheduler, 
                       train_dl : DataLoader, 
                       valid_dl : DataLoader, 
                       logger : LOGWRITER, 
                       loss_fn : CrossEntropyLoss, 
                       epochs : int, 
                       device='cuda'):
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
        device (str): The computation device ('cuda' or 'cpu').
    """
    best_loss = float('inf')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for images, labels in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        total_precision = 0
        total_recall = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for images, labels in tqdm(valid_dl, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total_precision += (preds == labels).sum().item() / preds.size(0)
                total_recall += (preds == labels).sum().item() / preds.size(0)
                total_accuracy += (preds == labels).sum().item() / preds.size(0)

            images, labels = next(iter(valid_dl))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            probabilities = F.softmax(outputs, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            
            cm = confusion_matrix(predictions=predictions,labels=labels,num_class=configs.num_class)
            plot_confusion_matrix(cm, configs.num_class, os.path.join(configs.matrix_output_dir, f"CONF_Matrix_{epoch+1}.png"))

        avg_train_loss = total_train_loss / len(train_dl)
        avg_val_loss = total_val_loss / len(valid_dl)
        avg_precision = total_precision / len(valid_dl)
        avg_recall = total_recall / len(valid_dl)
        avg_accuracy = total_accuracy / len(valid_dl)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(configs.save_pth, f'Best_model_CIFAR10_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)

        logger.write(epoch=epoch+1, tr_loss=avg_train_loss, val_loss=avg_val_loss,
                     precision=avg_precision, recall=avg_recall, accuracy=avg_accuracy)

        if epoch > configs.warm_up_epochs:
            scheduler.step()

def main():
    configs.main()

    train_dl = load_dataset(mode = "train")
    valid_dl = load_dataset(mode = "test")
    print(f"[INFO] Training Dataloader loaded with {len(train_dl)} batches.")
    print(f"[INFO] Validation Dataloader loaded with {len(valid_dl)} batches.")

    loss_fn = CrossEntropyLoss() 
    print("[INFO] Cross Entropy Function loaded.")

    if configs.model == "ViT":
        model = get_ViT(num_classes = configs.num_class)
        print("[INFO] ViT Model loaded with the following attributes:")
        print(f"[INFO] *: patch size: {model.patch_size}.")
        print(f"[INFO] *: num_layers: {model.layers}.")
        print(f"[INFO] *: d_model: {model.d_model}.")
        print(f"[INFO] *: number of attention heads: {model.head}.")
    elif configs.model == "ResNet18": 
        model = get_ResNet18(num_classes = configs.num_class)
        print("[INFO] ResNet18 Model loaded with the following attributs: ")
        print(f"[INFO] *: Channels: {model.channels}")
        print(f"[INFO] *: Layers: {model.num_layers}")
    elif configs.model == "ResNet34": 
        model = get_ResNet34(num_classes=configs.num_class)
        print("[INFO] ResNet34 Model loaded with the following attributs: ")
        print(f"[INFO] *: Channels: {model.channels}")
        print(f"[INFO] *: Layers: {model.num_layers}")
    else: 
        raise ValueError(f"[ERROR] Model not found or not confingured")

    if configs.model_save_path: 
        print("[INFO] Model weights provided. Loading model weights.")
        model.load_state_dict(torch.load(configs.model_save_path))
    
    optimizer = get_AdamW_optimizer(model=model, lr = configs.lr, weight_decay=configs.weight_decay)
    print(f"[INFO] Optimizer loaded with learning rate: {configs.lr}.")

    scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=configs.epochs-10, eta_min=1e-5, last_epoch=-1)
    print(f"[INFO] CosineAnnealingLR Scheduler loaded.")

    logger = LOGWRITER(output_directory=configs.log_output_dir, total_epochs=configs.epochs)
    print(f"[INFO] Log writer loaded and binded to {configs.log_output_dir}")
    print(f"[INFO] Total epochs: {configs.epochs}")
    print(f"[INFO] Warm Up Phase: {configs.warm_up_epochs} epochs")

    train_and_evaluate(model=model, optimizer=optimizer, scheduler=scheduler, train_dl=train_dl, valid_dl=valid_dl, logger=logger, loss_fn=loss_fn, epochs=configs.epochs)

if __name__ == "__main__": 
    main()