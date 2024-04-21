import torch
import torch.nn as nn 
from typing import Tuple, Union
import math
from utils.patches import * 


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
        Scaled dot product for calculating attention scores with Queries, Keys, Values 

        Args: 
            Queries (torch.Tensor): 
            Keys (torch.Tensor): 
            Values (torch.Tensor): 
            Mask (Union[None, torch.Tensor])
        
        Returns: 
            torch.Tensor
        """
        attn_score = torch.matmul(Queries, torch.transpose(Keys, -2, -1)) / math.sqrt(self.d_k) # Measures similarities between each set of queries and keys
        if Mask: 
            attn_scores = attn_scores.masked_fill(Mask == 0, -1e9)        
        QK_probs = torch.softmax(attn_score, dim = -1) # Scales the similarities between each query in Q to the entire set of Keys as probabilities
        output = torch.matmul(QK_probs, Values) # Transforms values into weighted sums, reflecting importance of each value within Values
        return output

    def forward(self, Queries, Keys, Values, Mask : Union[None, torch.Tensor] = None):
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
        x = x + self.l_norm_1(self.msa(x, x, x))
        x = x + self.l_norm_2(self.ffn(x))
        return x

class ViT(nn.Module): 
    def __init__(self, input_dim : Tuple[int] = (3,128,128), patch_size : int = 8, layers : int = 12, num_classes : int = 12): 
        super().__init__()

        self.d_model = input_dim[0] * patch_size ** 2
        self.n_patches = input_dim[1] * input_dim[2] // patch_size ** 2

        self.patch_embed = PatchEmbeddingConv(input_channels=input_dim[0], patch_size=patch_size)

        self.class_token = nn.Parameter(data=torch.randn(1, 1, self.d_model), requires_grad=True)

        self.pos_encod = nn.Parameter(data=torch.randn(1, self.n_patches + 1, self.d_model), requires_grad=True)

        self.encoder_stack = nn.Sequential(*[EncoderBlock(self.d_model, self.d_model * 4, self.d_model, 12, dropout=0.3) for _ in range(layers)])

        self.classifier_head = nn.Sequential(*[nn.LayerNorm(self.d_model),
                                               nn.Linear(self.d_model, num_classes)])
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x): 
        batch_size = x.size(0)

        patched_image = self.patch_embed(x)

        class_tokens = self.class_token.expand(batch_size, -1, -1)

        concat_patches = torch.cat([patched_image, class_tokens], dim = 1)

        x = self.encoder_stack(self.dropout(self.pos_encod + concat_patches))

        return self.classifier_head(x)

def main(): 
    batch_size = 1
    channels = 3 
    img_h = 128 
    img_w = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn((batch_size, channels, img_h, img_w), device = device)

    model = ViT().to(device)

    y = model(x)

    print(y.shape)

if __name__ == "__main__": 
    main()




