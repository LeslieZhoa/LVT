'''
@author: LeslieZhao
@Date: 20220712
'''
from model.base_model import Model
import numpy as np
from utils import utils 


class FaceGender(Model):
    def __init__(self, model_path):
        self.gender_detector = Model.__init__(self,model_path)
   
    def get_gender(self,img):
        outputs = self.gender_detector.run(None, {'input': img.astype(np.float32)})
        out_put_onnx = outputs[0]
        return out_put_onnx
    
    def preprocess_gender(self,img,align=False,lmk=None):
        align_lmk = None
        if align:
            align_img,align_lmk,_ = utils.align_face(img,lmk)
            img = align_img
        return utils.preprocess(img,[224,224],mean=self.lmk_mean,std=self.lmk_std),align_lmk

    def postprocess_gender(self,outputs):
        '''
        race_pred:
            0:White
            1:Black
            2:Latino_Hispanic
            3:East Asian
            4:Southeast Asian
            5:Indian
            6:Middle Eastern
        gender_pre:
            0:Male
            1:Female
        age_pred: 
            0:0-2
            1:3-9
            2:10-19
            3:20-29
            4:30:39
            5:40-49
            6:50-59
            7:60-69
            8:70+
        '''
        outputs = np.squeeze(outputs)
        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        return race_pred,gender_pred,age_pred
