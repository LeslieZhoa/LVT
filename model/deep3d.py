'''
@author: LeslieZhao
@Date: 20220712
'''
from model.base_model import Model
import numpy as np
from utils import utils 


class Deep3d(Model):
    def __init__(self,deep3d_path,bfm_path):
        self.deep3d_model = Model.__init__(self,deep3d_path)
           
        self.facemodel = Model.__init__(self,bfm_path)
    
    def get_deep3d_coeff(self,img):
        
        return self.deep3d_model.run(None, {'input': img.astype(np.float32)})[0]

    
    def preprocess_deep3d(self,x):
        return utils.preprocess(x,[224,224],
                    mean=None,
                    std=None)

    def postprocess_deep3d(self,coeff,lmks):
        _, _, _, _,projs = self.facemodel.run(None, {'input': coeff.astype(np.float32)})
        lmk_map = utils.convert_lmk_map(projs[0],lmks)
        return lmk_map