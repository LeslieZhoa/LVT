'''
@author: LeslieZhao
@Date: 20220712

'''
from cgi import test
import torch
from torch import nn

import numpy as np
from collections import OrderedDict
import sys 

import onnxruntime as ort
import onnx
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
    
    @staticmethod
    def get_input_names():
        return ['input']
    @staticmethod
    def get_output_names():
        return ['output']

    @staticmethod
    def get_dynamic_axes():
        return {'input':{0:'batch_size'},
                'output':{0:'batch_size'}}



def convert_model(Model,output_path,**kwargs):

    
    torch_model = Model(**kwargs)
    inputs = Model.get_input()

    torch_model(inputs)
    
    torch.onnx.export(torch_model,
                    inputs,
                    output_path,
                    export_params=True, 
                    opset_version=11, 
                    do_constant_folding=True,
                    input_names=Model.get_input_names(),
                    output_names=Model.get_output_names(),
                    dynamic_axes=Model.get_dynamic_axes()
                    )
   

def get_model(Model,model_path):
    model = onnx.load(model_path)
    # ort_session = ort.InferenceSession(model_path)
    ort_session = ort.InferenceSession(model.SerializeToString(),providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_names = Model.get_input_names()
    return ort_session,input_names

def get_feed_data(input_names,data):
    if len(input_names) == 1:
        data = [data]
    feed_dict = {x:y for x,y in zip(input_names,data)}
    return feed_dict

def test_model(Model,model_path):

    ort_session,input_names = get_model(Model,model_path)
    
    inputs = Model.get_input()
    input_names = Model.get_input_names()
    if len(input_names) != 1:
        inputs = [to_numpy(x) for x in inputs]
    else:
        inputs = to_numpy(inputs)
    
    feed_dict = get_feed_data(input_names,inputs)

    outputs = ort_session.run(None,feed_dict)

    return outputs
    

if __name__ == "__main__":
    
    convert_model(SLPT,'slpt-lmk.onnx',load_model_path='/mnt/user/zhaoxiang/workspace/BaseModel/SLPT-master/Weight/WFLW_6_layer.pth')
    test_model(SLPT,'slpt-lmk.onnx')