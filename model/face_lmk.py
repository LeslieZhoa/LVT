'''
@author: LeslieZhao
@Date: 20220712
'''
from model.base_model import Model
import numpy as np
import cv2
from utils import utils 


class FaceLmk(Model):
    def __init__(self,model_path):
        
        self.face_lmk = Model.__init__(self,model_path)
        
    def get_lmk(self,img):
        return self.face_lmk.run(None, {'input': img.astype(np.float32)})[0]

    def preprocess_lmk(self,img):
        
        return utils.preprocess(img,(256,256),None,None)
       
    def postprocess_lmk(self,lmk,height,top):
        lmk = lmk * height
        lmk = lmk + np.array(top).reshape(-1,2)
        return lmk
        
        