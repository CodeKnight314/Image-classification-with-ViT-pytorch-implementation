import torch 
import torch.nn as nn 
import time 
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        N, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "[ERROR] Image's Height or Width is not divisible by the patch size"
        
        x = x.view(N, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        
        x = x.view(N, C * self.patch_size**2, -1)

        x = x.permute(0, 2, 1)

        return x
    
class PatchEmbeddingConv(nn.Module): 
    def __init__(self, input_channels : int = 3, patch_size : int = 16): 
        super().__init__()

        self.input_channels = input_channels 
        self.patch_size = patch_size 

        self.in_conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels * patch_size**2, kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x): 
        x = self.flatten(self.in_conv(x))
        return x.permute(0, 2, 1)

def main(): 
    p_embed = PatchEmbedding(16)
    arr = []
    for i in range(1, 9): 
        x = torch.rand((2**i,3,224,224), dtype = torch.float32, device = "cuda" if torch.cuda.is_available() else "cpu")
        start = time.time()
        y = p_embed(x)
        end = time.time() - start
        arr.append(end)
        print(round(end, 4), 2**i, y.shape)

    plt.scatter([2**i for i in range(1, len(arr)+1)], arr)
    plt.title("Standard Patch Embedding")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (seconds)")
    plt.show()

    p_embed = PatchEmbeddingConv().to(device)
    arr = [] 
    for i in range(1, 9): 
        x = torch.rand((2**i, 3, 224, 224), dtype = torch.float32, device = "cuda" if torch.cuda.is_available() else "cpu")
        start = time.time() 
        y = p_embed(x)
        end = time.time() - start
        arr.append(end)
        print(round(end, 4), 2**i, y.shape)

    plt.scatter([2**i for i in range(1, len(arr)+1)], arr)
    plt.title("Conv Based embedding")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (seconds)")
    plt.show()
    
if __name__ == "__main__": 
    main() 