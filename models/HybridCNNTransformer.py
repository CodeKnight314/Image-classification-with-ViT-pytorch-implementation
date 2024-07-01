import torch 
import torch.nn as nn 
from typing import Tuple
from ViT import EncoderBlock, PatchEmbeddingConv

class CNNBackbone(nn.Module): 
    def __init__(self, input_channels, output_size = (128, 128)):
        super().__init__()

        self.conv_1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pooling = nn.AdaptiveAvgPool2d(output_size=output_size)

        self.relu = nn.ReLU()

    def foward(self, x): 
        out1 = self.conv_1(x)
        out1 = self.relu(out1)
        out2 = self.conv_2(x)
        out2 = self.relu(out2)
        out3 = self.conv_3(x)
        out3 = self.relu(out3)

        output = self.pooling(out3)

        return output

class HybridCNNTransformer(nn.Module): 
    def __init__(self, input_channels : int, cnn_output_size : Tuple[int, int], d_model : int, patch_size : int, head : int, num_layers : int, num_classes : int):
        super().__init__() 

        self.d_model = d_model
        self.patch_size = patch_size
        self.num_head = head

        self.cnn_backbone = CNNBackbone(input_channels=input_channels, output_size=cnn_output_size)
        
        self.patch_embed = PatchEmbeddingConv(input_channels=256, patch_size=patch_size, d_model=self.d_model)

        self.class_token = nn.Parameter(data=torch.randn(1, 1, self.d_model), requires_grad=True)
        self.init_pos_encod(cnn_output_size)

        self.encoder_stack = nn.Sequential(*[EncoderBlock(self.d_model, self.d_model * 4, self.d_model, head=head, dropout=0.1) for _ in range(num_layers)])

        self.classifier = nn.Sequential(*[nn.LayerNorm(self.d_model),
                                          nn.Linear(self.d_model, num_classes),
                                          nn.Dropout(0.1)])
        
    def init_pos_encod(self, cnn_output_shape):
        """
        Initializes position encoding for the encoder stack.
        """
        n_patches = (cnn_output_shape[0] // self.patch_size) * (cnn_output_shape[1] // self.patch_size)
        self.pos_encod = nn.Parameter(data=torch.randn(1, n_patches + 1, self.d_model), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        cnn_features = self.cnn_backbone(x)
        
        patched_image = self.patch_embed(cnn_features)
        
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        concat_patches = torch.cat([class_tokens, patched_image], dim=1)
        
        x = self.pos_encod + concat_patches
        
        x = self.encoder_stack(x)

        return self.classifier(x[:, 0, :])