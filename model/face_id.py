'''
@author: LeslieZhao
@Date: 20220712
'''
from model.base_model import Model
import numpy as np
from utils import utils 


class FaceID(Model):
    def __init__(self,model_path):
        self.id_detector = Model.__init__(self,model_path)
    
    def get_id(self,img):
        return self.id_detector.run(None, {'input': img.astype(np.float32)})[0]

    
    def preprocess_faceid(self,x):
        return utils.preprocess(x[28:228,28:228],size=[112,112],mean=None,std=None)
