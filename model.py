import torch
import torch.nn as nn 
from typing import Tuple, Union
import math
from utils.patches import * 
import matplotlib.pyplot as plt
from tqdm import tqdm 
from torchsummary import summary
import configs 

class MSA(nn.Module): 
    def __init__(self, d_model : int, head : int):
        assert d_model % head == 0, f"[Error] d_model {d_model} is not divisible by head {head} in MSA Module"
        super().__init__()

        self.d_model = d_model 
        self.head = head
        self.d_k = d_model // head 

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.head)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.head)
        self.W_V = nn.Linear(self.d_model, self.d_k * self.head)
        self.W_O = nn.Linear(self.d_k * self.head, self.d_model)

    def scaled_dot_product(self, Queries, Keys, Values, Mask : Union[None, torch.Tensor] = None): 
        """
        Computes the scaled dot-product attention over the inputs.

        Args:
            Queries, Keys, Values (torch.Tensor): The query, key, and value tensors after being processed through
                their respective linear transformations.
            Mask (torch.Tensor, optional): Optional mask tensor to zero out certain positions in the attention scores.

        Returns:
            torch.Tensor: The output tensor after applying attention and weighted sum operations.
        """
        attn_score = torch.matmul(Queries, torch.transpose(Keys, -2, -1)) / math.sqrt(self.d_k) # Measures similarities between each set of queries and keys
        if Mask: 
            attn_scores = attn_scores.masked_fill(Mask == 0, -1e9)        
        QK_probs = torch.softmax(attn_score, dim = -1) # Scales the similarities between each query in Q to the entire set of Keys as probabilities
        output = torch.matmul(QK_probs, Values) # Transforms values into weighted sums, reflecting importance of each value within Values
        return output

    def forward(self, Queries, Keys, Values, Mask : Union[None, torch.Tensor] = None):
        """
        Forward pass of the MSA module. Applies self-attention individually to each head, concatenates the results,
        and then projects the concatenated output back to the original dimensionality.

        Args:
            Queries, Keys, Values (torch.Tensor): Inputs to the self-attention mechanism.
            Mask (torch.Tensor, optional): Optional mask to apply during the attention mechanism.

        Returns:
            torch.Tensor: The output of the MSA module after processing through the attention mechanism and linear output layer.
        """
        batch_size = Queries.size(0)

        Q = self.W_Q(Queries).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        K = self.W_K(Keys).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        V = self.W_V(Values).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        scaled_values = self.scaled_dot_product(Queries=Q, Keys=K, Values=V, Mask=Mask).transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.head)

        context = self.W_O(scaled_values)

        return context

class FFN(nn.Module): 
    def __init__(self, input_dim : int, hidden_dim : int, output_dim : int):
        super().__init__() 

        self.l_1 = nn.Linear(input_dim, hidden_dim)
        self.l_2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x): 
        """
        Forward pass through the feed-forward network. Applies a linear transformation followed by a ReLU activation
        and another linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch size, feature dimension].

        Returns:
            torch.Tensor: Output tensor of shape [batch size, output dimension].
        """
        return self.l_2(self.relu(self.l_1(x)))
    
class EncoderBlock(nn.Module): 
    def __init__(self, input_dim, hidden_dim, outuput_dim, head, dropout): 
        super().__init__()

        self.msa = MSA(d_model=input_dim, head=head)
        self.ffn = FFN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=outuput_dim)
        self.l_norm_1 = nn.LayerNorm(input_dim)
        self.l_norm_2 = nn.LayerNorm(outuput_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Processes input through the encoder block, applying multi-head self-attention, feed-forward network, 
        and residual connections with layer normalization.

        Args:
            x (torch.Tensor): The input tensor to the encoder block.

        Returns:
            torch.Tensor: The output tensor from the encoder block after processing.
        """ 
        x = x + self.l_norm_1(self.msa(x, x, x))
        x = x + self.l_norm_2(self.ffn(x))
        return x

class ViT(nn.Module):
    def __init__(self, input_dim=(3, 320, 320), patch_size=8, layers=12, num_classes=12):
        super().__init__()

        self.d_model = min(input_dim[0] * patch_size ** 2, 512)
        self.patch_size = patch_size
        self.input_dim = input_dim

        self.patch_embed = PatchEmbeddingConv(input_channels=input_dim[0], patch_size=patch_size)

        self.class_token = nn.Parameter(data=torch.randn(1, 1, self.d_model), requires_grad=True)

        self.init_pos_encod()

        self.encoder_stack = nn.Sequential(*[EncoderBlock(self.d_model, self.d_model, self.d_model, 12, dropout=0.3) for _ in range(layers)])

        self.classifier_head = nn.Sequential(*[nn.LayerNorm(self.d_model),
                                               nn.Linear(self.d_model, num_classes),
                                               nn.Dropout(0.3)])

    def init_pos_encod(self):
        n_patches = (self.input_dim[1] // self.patch_size) * (self.input_dim[2] // self.patch_size)
        self.pos_encod = nn.Parameter(data=torch.randn(1, n_patches + 1, self.d_model), requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        patched_image = self.patch_embed(x)
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        concat_patches = torch.cat([patched_image, class_tokens], dim=1)

        x = self.encoder_stack(self.pos_encod + concat_patches)

        return self.classifier_head(x[:, 0, :])
    
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
    
def get_ViT(input_dim: Tuple[int] = (3, configs.img_height, configs.img_width), patch_size=configs.patches, layers: int = 12, device: str = configs.device):
    return ViT(input_dim=input_dim, patch_size=patch_size, layers=layers, num_classes=configs.num_class).to(device)

def get_ResNet(channels=[64, 128, 256, 512], num_layers=[3, 4, 6, 3], num_classes: int = configs.num_class, device: str = configs.device):
    return ResNet(channels=channels, num_layers=num_layers, num_classes=num_classes).to(device)
    
def main():  

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arr = []
    for i in tqdm(range(1,6)):
        runtime_dim = []
        for j in range(3):
            batch_size = 2**i
            channels = 3 
            img_h = 64 * 2**j
            img_w = 64 * 2**j

            model = ViT((channels, img_h, img_w)).to(device)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            x = torch.randn((batch_size, channels, img_h, img_w), device = device)
            
            start = time.time()
            y = model(x)
            end = time.time() - start

            runtime_dim.append(end)
        
        arr.append(runtime_dim)

        plt.scatter([2**i for i in range(len(runtime_dim))], runtime_dim)    
        plt.title("Run Time of Model")
        plt.xlabel("Batch Size")
        plt.ylabel("Runtime (seconds)")
        plt.savefig(f"Model_{2**i}.png")
    
    model = ViT((3, 256, 256)).to(device)
    summary(model, (3, 256, 256))

if __name__ == "__main__": 
    main()