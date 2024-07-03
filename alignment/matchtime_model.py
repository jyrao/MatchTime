import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import clip


class VideoEncoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=128):
        super(VideoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 384)  
        self.bn1 = nn.BatchNorm1d(384)
        self.fc2 = nn.Linear(384, 256)  
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, output_dim) 
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        bs = x.shape[0]
        x = rearrange(x, 'b l a f -> (b l) (a f)') 
        x = F.leaky_relu(self.bn1(self.fc1(x)))  
        x = F.leaky_relu(self.bn2(self.fc2(x)))  
        x = self.bn3(self.fc3(x)) 
        x = rearrange(x, '(b l) f -> b l f', b=bs) 
        return x
    
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.fc1 = nn.Linear(512, 384)
        self.bn1 = nn.BatchNorm1d(384)
        self.fc2 = nn.Linear(384, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x

class ContrastiveLearningModel(nn.Module):
    def __init__(self, feature_dim=512, embedding_dim=128, device="cuda:7"):
        super(ContrastiveLearningModel, self).__init__()
        self.video_encoder = VideoEncoder(input_dim=feature_dim, output_dim=embedding_dim).to(device=device, dtype=torch.bfloat16)
        self.text_encoder = TextEncoder().to(device=device, dtype=torch.bfloat16)
        self.model, _ = clip.load("ViT-B/32", device=device)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
    def forward(self, anchor_caption, concat_feature):
        anchor_embeddings = self.model.encode_text(anchor_caption).to(torch.bfloat16)
        anchor_encoded = self.text_encoder(anchor_embeddings).unsqueeze(2)
        concat_encoded = self.video_encoder(concat_feature)

        logits = torch.bmm(concat_encoded, anchor_encoded).squeeze(2)
        labels = torch.zeros(anchor_embeddings.shape[0], dtype=torch.long).to(device=anchor_embeddings.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss, logits