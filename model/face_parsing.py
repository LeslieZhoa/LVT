'''
@author: LeslieZhao
@Date: 20220712
'''
import numpy as np
import cv2
import torch
from utils import utils 


class FaceParsing:
    def __init__(self,model_path):
        
        self.parsing_model = torch.jit.load(model_path)
        if torch.cuda.is_available():
            self.parsing_model = self.parsing_model.cuda()
        self.parsing_model.eval()
        self.lmk_mean =np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.lmk_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.lmk_image_size=[256,256]
    
    def get_parsing(self,img):
        '''
        atts = ['background','skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        '''
        return self.parsing_model(img)

    def preprocess_parsing(self,x):
        x = utils.preprocess(x,size=[512,512],mean=self.lmk_mean,std=self.lmk_std)
        x = torch.from_numpy(x.astype(np.float32))
        if torch.cuda.is_available():
            x = x.cuda()
        return x 

    def postprocess_parsing(self,y,h=None,w=None):
        y = y[0].argmax(0).cpu().numpy()
        if h is not None:
            y = cv2.resize(y.astype(np.uint8),[w,h],interpolation=cv2.INTER_NEAREST)
        return y
