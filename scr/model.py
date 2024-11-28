import torch
import torch.nn as nn
from .sbert import vectorize as vec

class Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(0.2),
        )
    
    def forward(self, x):
        return self.model(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Block(1024, 512),
            nn.LeakyReLU(),
            
            Block(512, 256),
            nn.LeakyReLU(),
            
            Block(256, 128),
            nn.LeakyReLU(),
            
            Block(128, 64),
            nn.LeakyReLU(),
            
            Block(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, text):
        return self(vec(text))