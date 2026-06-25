import torch 
import torch.nn as nn 
import yaml
from torchvision.models import resnet50, ResNet50_Weights 

class Custom_cnn(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        layers = []
        current_channel = 3

        for hidden_dim in model_config['hidden_layers']:
            layers.append(nn.Conv2d(current_channel, hidden_dim, kernel_size =3, padding =1))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            current_channel = hidden_dim 

        self.cnn = nn.Sequential(*layers)
        
        final_spatial = model_config['input_size'] // ( 2 ** len(model_config['hidden_layers']))
        flattened_size = current_channel * final_spatial * final_spatial
        self.fc  = nn.Sequential(nn.Linear(flattened_size, 256),
                                 nn.ReLU(),
                                 nn.Dropout(model_config['dropout']),
                                 nn.Linear(256, model_config['num_classes']))
        
    def forward(self,x):
        x = self.cnn(x)
        x= x.flatten(1)
        return self.fc(x)

def server_resnet50(model_cfg):
    weights = ResNet50_Weights.DEFAULT if model_cfg['pretrained'] else None
    model = resnet50(weights=weights)
    
    if model_cfg['freeze_backbone']:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, model_cfg['num_classes'])
    return model
