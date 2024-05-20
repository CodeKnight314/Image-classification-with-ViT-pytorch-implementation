import torch 
import torch.nn as nn 
import configs 
import optuna 
from dataset import * 
from loss import * 
from train import train_step
from eval import eval_step
from tqdm import tqdm 

class ResBlock(nn.Module): 
    def __init__(self, input_channels: int, output_channels: int, stride: int): 
        super().__init__()

        if input_channels != output_channels: 
            self.projection = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(output_channels)
            )
        else: 
            self.projection = nn.Identity()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.ba_n1 = nn.BatchNorm2d(output_channels)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.ba_n2 = nn.BatchNorm2d(output_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.projection(x)
        
        out = self.relu(self.ba_n1(self.conv1(x)))
        out = self.ba_n2(self.conv2(out))
        out = self.relu(out)  
        
        out = out + identity
        out = self.relu(out) 
    
        return out

class ResStack(nn.Module): 
    def __init__(self, input_channels : int, output_channels : int, num_layers : int, stride : int = 1): 
        super().__init__() 

        layers = []
        layers.append(ResBlock(input_channels=input_channels, output_channels=output_channels, stride=stride))
        for _ in range(num_layers - 1):
            layers.append(ResBlock(input_channels=output_channels, output_channels=output_channels, stride=1))
        self.block = nn.Sequential(*layers)

    def forward(self, x): 
        return self.block(x)

class ResNet(nn.Module): 

    def __init__(self, channels = [64, 128, 256, 512], num_layers = [3, 4, 6, 3], num_classes : int = configs.num_class):
        super().__init__()

        assert len(channels) == len(num_layers), "[ERROR] Channels and Layers lists do not match in length."

        self.input_conv = nn.Sequential(*[nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=7, stride=2, padding=3, bias=False),
                                          nn.BatchNorm2d(64), 
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=3, stride=2)])
        
        blocks = []
        blocks.append(ResStack(64, channels[0],num_layers[0], 1))
        for i in range(1, len(channels)):
            blocks.append(ResStack(channels[i-1], channels[i], num_layers[i], 2))

        self.blocks = nn.Sequential(*blocks)

        self.classifier_head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                            nn.Flatten(), 
                                            nn.Linear(channels[-1], num_classes), 
                                            nn.Dropout(0.1))

    def forward(self, x): 
        first_conv = self.input_conv(x)
        block_conv = self.blocks(first_conv)
        logits = self.classifier_head(block_conv)

        return logits
    
def get_ResNet(channels=[64, 128, 256, 512], num_layers=[3, 4, 6, 3], num_classes: int = configs.num_class, device: str = configs.device):
    return ResNet(channels=channels, num_layers=num_layers, num_classes=num_classes).to(device)

def objective_resnet(trial):
    channels = trial.suggest_categorical('channels', [[64, 128, 256, 512], [64, 128, 256, 1024]])
    num_layers = trial.suggest_categorical('num_layers', [[3, 4, 6, 3], [3, 4, 23, 3]])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-1)

    model = ResNet(channels=channels, num_layers=num_layers, num_classes=configs.num_class).to(configs.device)
    optimizer = get_optimizer(model, lr=lr, betas=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = load_dataset()
    test_loader = load_dataset()

    for epoch in tqdm(range(10), desc=f'Trial {trial.number+1}', unit='epoch'):
        batched_values = []
        for i, data in enumerate(train_loader):
            loss = train_step(model, optimizer, data, criterion)
            batched_values.append(loss)

        averaged_values = torch.tensor(batched_values).mean().item()

    batched_values = []
    for i, data in enumerate(test_loader):
        loss, accuracy = eval_step(model, data, criterion, device=configs.device)
        batched_values.append([loss, accuracy])

    averaged_values = torch.tensor(batched_values).mean(dim=0)
    return averaged_values[1]  

if __name__ == '__main__':
    study_resnet = optuna.create_study(direction='maximize')
    n_trials = 50 
    study_resnet.optimize(objective_resnet, n_trials=n_trials)

    print('Best trial for ResNet:')
    print(study_resnet.best_trial)