import yaml 
from src.Models import Custom_cnn, server_resnet50
import torch
import os

def yaml_loader(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def model_registry(config):
    selected_model = config['model_selection']
    if selected_model == 'custom_cnn':
        model = Custom_cnn(config['custom_cnn'])
        return model, config[selected_model]['input_size']
    elif selected_model == 'server_resnet50':
        model = server_resnet50(config['server_resnet50'])
        return model, config[selected_model]['input_size']
    else:
        raise ValueError(f"Model '{selected_model}' not found in registry.")
    
def load_checkpoint(config):
    checkpoint_pathname = config['model_weight']['model_path']
    root = os.getcwd()
    checkpoint_path = root + checkpoint_pathname
    checkpoint = torch.load(checkpoint_path,map_location='cpu')
    return checkpoint
