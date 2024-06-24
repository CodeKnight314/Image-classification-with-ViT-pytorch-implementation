import argparse
from models.ViT import get_ViT
from models.ResNet import get_ResNet18, get_ResNet34
from dataset import load_dataset
from loss import CrossEntropyLoss
from utils.visualization import plot_confusion_matrix, confusion_matrix
from utils.log_writer import LOGWRITER
import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import configs
from sklearn.metrics import precision_score, recall_score, accuracy_score

def train_and_evaluate(model, optimizer, scheduler, train_dl, valid_dl, logger, loss_fn, epochs, device='cuda'):
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
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels in tqdm(valid_dl, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)
        
        avg_train_loss = total_train_loss / len(train_dl)
        avg_val_loss = total_val_loss / len(valid_dl)

        images, labels = next(iter(valid_dl))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        probabilities = torch.nn.functional.softmax(outputs, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
        cm = confusion_matrix(predictions=predictions, labels=labels, num_class=configs.num_class)
        plot_confusion_matrix(cm, configs.num_class, os.path.join(configs.matrix_output_dir, f"CONF_Matrix_{epoch+1}.png"))

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(configs.save_pth, f'Best_model_CIFAR10_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)

        logger.write(epoch=epoch+1, tr_loss=avg_train_loss, val_loss=avg_val_loss,
                     precision=precision, recall=recall, accuracy=accuracy)

        if epoch > configs.warm_up_epochs:
            scheduler.step()

def main():
    parser = argparse.ArgumentParser(description='Train a model on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs to train')
    parser.add_argument('--model', type=str, required=True, choices=['ViT', 'ResNet18', 'ResNet34'], help='Model name')
    parser.add_argument('--patch_size', type=int, help="ViT Patch Size")
    parser.add_argument('--model_save_path', type=str, help='Path to save or load model weights')
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory to Dataset. Must contain a train and test folder in root directory.")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'StepLR'], help='Learning rate scheduler')
    parser.add_argument('--t_max', type=int, default=80, help='T_max for CosineAnnealingLR')
    parser.add_argument('--eta_min', type=float, default=1e-5, help='Eta_min for CosineAnnealingLR')
    parser.add_argument('--step_size', type=int, default=30, help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')

    args = parser.parse_args()

    train_dl = load_dataset(root_dir=args.root_dir, mode="train")
    valid_dl = load_dataset(root_dir=args.root_dir, mode="test")
    print(f"[INFO] Training Dataloader loaded with {len(train_dl)} batches.")
    print(f"[INFO] Validation Dataloader loaded with {len(valid_dl)} batches.")

    loss_fn = CrossEntropyLoss()
    print("[INFO] Cross Entropy Function loaded.")

    if args.model == "ViT":
        model = get_ViT(patch_size=args.patch_size, num_classes=configs.num_class)
        print("[INFO] ViT Model loaded with the following attributes:")
        print(f"[INFO] * Patch size: {model.patch_size}.")
        print(f"[INFO] * Number of layers: {model.layers}.")
        print(f"[INFO] * Model dimension: {model.d_model}.")
        print(f"[INFO] * Number of attention heads: {model.head}.")
    elif args.model == "ResNet18":
        model = get_ResNet18(num_classes=configs.num_class)
        print("[INFO] ResNet18 Model loaded with the following attributes:")
        print(f"[INFO] * Channels: {model.channels}")
        print(f"[INFO] * Layers: {model.num_layers}")
    elif args.model == "ResNet34":
        model = get_ResNet34(num_classes=configs.num_class)
        print("[INFO] ResNet34 Model loaded with the following attributes:")
        print(f"[INFO] * Channels: {model.channels}")
        print(f"[INFO] * Layers: {model.num_layers}")

    if args.model_save_path:
        print("[INFO] Model weights provided. Loading model weights.")
        model.load_state_dict(torch.load(args.model_save_path))
    
    if args.optimizer == 'AdamW':
        optimizer = opt.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = opt.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    print(f"[INFO] Optimizer loaded with learning rate: {args.lr}.")

    if args.scheduler == 'CosineAnnealingLR':
        scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)
    elif args.scheduler == 'StepLR':
        scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print(f"[INFO] {args.scheduler} Scheduler loaded.")

    logger = LOGWRITER(output_directory=configs.log_output_dir, total_epochs=args.epochs)
    print(f"[INFO] Log writer loaded and binded to {configs.log_output_dir}")
    print(f"[INFO] Total epochs: {args.epochs}")
    print(f"[INFO] Warm Up Phase: {configs.warm_up_epochs} epochs")

    configs.main()

    train_and_evaluate(model=model, 
                       optimizer=optimizer, 
                       scheduler=scheduler, 
                       train_dl=train_dl, 
                       valid_dl=valid_dl, 
                       logger=logger, 
                       loss_fn=loss_fn, 
                       epochs=args.epochs)

if __name__ == "__main__":
    main()
