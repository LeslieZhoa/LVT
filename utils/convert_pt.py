from turtle import forward
import torch
from torch import nn

import numpy as np
from collections import OrderedDict
import sys 

import pdb

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class BaseModel(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        if 'load_model_path' in kwargs:
            load_model_path = kwargs['load_model_path']
            self.pre_state_dict = torch.load(load_model_path, map_location='cpu')
       
    def forward(self,x):
        pass

    @staticmethod
    def get_input(batch_size=1,img_size=256):
        x = torch.randn(batch_size,3,img_size,img_size,requires_grad=True)
        return x
    
    



    
    
def convert_model(Model,output_path,**kwargs):

    
    torch_model = Model(**kwargs)
    inputs = Model.get_input()
    
    traced_script_module = torch.jit.trace(torch_model, inputs)
    traced_script_module.save(output_path)   
   

def get_model(Model,model_path):
    model = torch.jit.load(model_path)
    # ort_session = ort.InferenceSession(model_path)
    model.eval()
    return model

def get_feed_data(input_names,data):
    if len(input_names) == 1:
        data = [data]
    feed_dict = {x:y for x,y in zip(input_names,data)}
    return feed_dict

def test_model(Model,model_path):

    model = get_model(Model,model_path)
    inputs = Model.get_input()
    
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
    outputs = model(inputs)
    return outputs
    

if __name__ == "__main__":
   
    convert_model('',"",
            load_model_path='')
    
    test_model('',"*.pt") 
   