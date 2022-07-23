'''
@author: LeslieZhao
@Date: 20220712
'''
from model.base_model import Model
import numpy as np
import cv2
from utils import utils 


class FaceDetector(Model):
    def __init__(self,model_path):
        
        self.face_detector = Model.__init__(self,model_path)
        self.priors_cache = {}
    
    def get_face(self,img):
        loc, conf, iou = self.face_detector.run(None, 
                {self.face_detector.get_inputs()[0].name: img})

        return loc,conf,iou

    def preprocess_face(self,ori_img,small=False):
        img = ori_img.copy()
        
        if small:
            h,w,_ = ori_img.shape
            scale = min(512.0/h,512.0/w)
            if scale < 1:
                img = cv2.resize(img,None,fx=scale,fy=scale)
        if self.priors_cache.get(img.shape[:2], None) is None:
            priors = utils.anchor_fn(img.shape[:2])
            self.priors_cache[img.shape[:2]] = priors
        else:
            priors = self.priors_cache[img.shape[:2]]
        img = np.transpose(img, [2, 0, 1]).astype(np.float32)[np.newaxis, ...].copy()
        self.priors = priors
        return img

    def postprocess_face(self,origin_image,loc,conf,iou,
                        mx=1,score_thresh=0.3,nms_thresh=0.45):

        
        conf = utils.softmax(conf.squeeze(0))
        boxes = utils.box_decode(loc.squeeze(0), 
                    self.priors, variances=[0.1, 0.2])
        h,w,_ = origin_image.shape
        boxes[:, 0::2] *= w
        boxes[:, 1::2] *= h
        cls_scores = conf[:, 1]
        iou_scores = iou.squeeze(0)[:, 0]
        iou_scores = np.clip(iou_scores, a_min=0., a_max=1.)
        scores = np.sqrt(cls_scores * iou_scores)
        score_mask = scores > score_thresh
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        pre_det = np.hstack((boxes[:, :4], scores[:, None]))
        keep = utils.nms(pre_det, nms_thresh)

        kpss = boxes[keep, 4:]
        boxes = pre_det[keep, :]
        boxes = boxes[:mx]
        kpss = kpss[:mx]
        return boxes,kpss