import torch 
import torch.nn as nn 
from typing import Union, Tuple
import math 
import configs 
from loss import *
from dataset import *
import optuna
from tqdm import tqdm

class PatchEmbeddingConv(nn.Module):
    def __init__(self, input_channels: int = 3, patch_size: int = 8, d_model: int = 512):
        super().__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size

        self.in_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        """
        Forward pass that uses a convolutional layer to extract and flatten image patches.

        Args:
            x (torch.Tensor): A batch of images, expected shape (N, C, H, W) where
                              N is the batch size, C is the number of channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: The output tensor with patches flattened and arranged, shape (N, L, D) where
                          L is the number of patches and D is the dimension of each patch.
        """
        x = self.in_conv(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        return x

class MSA(nn.Module):
    def __init__(self, d_model : int, head : int, dropout : float = 0.3):
        assert d_model % head == 0, f"[Error] d_model {d_model} is not divisible by head {head} in MSA Module"
        super().__init__()

        self.d_model = d_model
        self.head = head
        self.d_k = d_model // head

        self.dropout = nn.Dropout(dropout)

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
        QK_probs = self.dropout(QK_probs)
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

        context = self.dropout(self.W_O(scaled_values))

        return context

class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.3):
        super().__init__()

        self.l_1 = nn.Linear(input_dim, hidden_dim)
        self.l_2 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the feed-forward network. Applies a linear transformation followed by a GELU activation,
        a dropout layer, and another linear transformation followed by GELU activation and dropout.

        Args:
            x (torch.Tensor): Input tensor of shape [batch size, feature dimension].

        Returns:
            torch.Tensor: Output tensor of shape [batch size, output dimension].
        """
        x = self.l_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.l_2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x

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
    def __init__(self, input_dim=(3, 64, 64), patch_size=8, layers=12, d_model=512, head=12, num_classes=12):
        super().__init__()

        self.d_model = d_model
        self.head = head
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.layers = layers
        self.dropout = 0.3

        self.patch_embed = PatchEmbeddingConv(input_channels=input_dim[0], patch_size=patch_size, d_model=self.d_model)

        self.class_token = nn.Parameter(data=torch.randn(1, 1, self.d_model), requires_grad=True)

        self.init_pos_encod()

        self.encoder_stack = nn.Sequential(*[EncoderBlock(self.d_model, self.d_model, self.d_model, 4, dropout=self.dropout) for _ in range(layers)])

        self.classifier_head = nn.Sequential(*[nn.LayerNorm(self.d_model),
                                               nn.Linear(self.d_model, num_classes),
                                               nn.Dropout(self.dropout)])

        self._initialize_weights()

    def init_pos_encod(self):
        """
        Initializes position encoding for the encoder stack.
        """
        n_patches = (self.input_dim[1] // self.patch_size) * (self.input_dim[2] // self.patch_size)
        self.pos_encod = nn.Parameter(data=torch.randn(1, n_patches + 1, self.d_model), requires_grad=True)

    def _initialize_weights(self):
        """
        Initializes weights for Conv2d, Linear, class tokens and positional encoding.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.class_token, mean=0, std=1)
        nn.init.normal_(self.pos_encod, mean=0, std=1)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        patched_image = self.patch_embed(x)
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        concat_patches = torch.cat([patched_image, class_tokens], dim=1)

        x = self.encoder_stack(self.pos_encod + concat_patches)

        return self.classifier_head(x[:, 0, :])
    
def get_ViT(input_dim: Tuple[int] = (3, configs.img_height, configs.img_width), 
            patch_size=8, 
            layers: int = 12, 
            d_model : int = 512, 
            head : int = 8,
            num_classes = configs.num_class, device: str = configs.device) -> ViT:
    """
    Helper function for getting VIT with configured parameters.

    Returns: 
        ViT: ViT model with configured parameters
    """
    
    return ViT(input_dim=input_dim, patch_size=patch_size, layers=layers, d_model=d_model, head=head, num_classes=num_classes).to(device)