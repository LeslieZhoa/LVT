'''
@author: LeslieZhao
@Date: 20220712
'''
from model import *
import cv2
from utils import utils 
import pdb

class Engine(FaceDetector,
            FaceParsing,
            FaceGender,
            FaceID,
            Deep3d,
            FaceLmk):
    def __init__(self,face_detector_path=None,
                     face_lmk_path=None,
                      gender_path=None,
                      deep3d_path=None,
                      bfm_path=None,
                      face_id_path=None,
                      face_parsing_path=None):

        
        if face_detector_path is not None:
            FaceDetector.__init__(self,face_detector_path)
            
        if face_lmk_path is not None:
            FaceLmk.__init__(self,face_lmk_path)
        #     self.face_lmk_detector = self.face_lmk_detector = ort.InferenceSession(
        #                     onnx.load(face_lmk_path).SerializeToString(),
        #                     providers=[
        #                     'CUDAExecutionProvider',
        #                     'CPUExecutionProvider'])

        if gender_path is not None:
        
            FaceGender.__init__(self,gender_path)

        if deep3d_path is not None:
            Deep3d.__init__(self,deep3d_path,bfm_path)

        if face_id_path is not None:
            FaceID.__init__(self,face_id_path)

        if face_parsing_path is not None:
            
            FaceParsing.__init__(self,face_parsing_path)

    

if __name__ == "__main__":
    
    img = cv2.imread('images/barack-obama-gty-jt-210802_1627927668233_hpMain_16x9_1600.jpeg')
    engine = Engine(face_detector_path='weights/yunet_final_dynamic_simplify.onnx'
                    ,face_lmk_path='weights/slpt-lmk.onnx'
                    ,gender_path='weights/fairface.onnx'
                    ,deep3d_path='weights/deep3d.onnx'
                    ,bfm_path='weights/postdeep3d.onnx'
                    ,face_id_path='weights/id_model.onnx'
                    ,face_parsing_path='weights/face_parsing.pt')
    
    # detect face
    p_img = engine.preprocess_face(img,True)
    loc,conf,iou = engine.get_face(p_img)

    # only select first box
    bboxes,kpss = engine.postprocess_face(img,loc,conf,iou)

    # lmk
    crop_img,top,crop_height = utils.crop_image(img,bboxes[0])
    inp = engine.preprocess_lmk(crop_img)
    lmks = engine.get_lmk(inp)
    lmks = engine.postprocess_lmk(lmks,crop_height,top)

    cv2.imwrite('crop.png',crop_img)
    cv2.imwrite('lmk.png',utils.draw_lmk(img,lmks[0]))

    # align and crop
    info = utils.align_crop_img(img,lmks[0])
    align_face = info['face']
    cv2.imwrite('align.png',align_face)

    # get parsing 
    
    p_img = engine.preprocess_parsing(crop_img)

    parsing = engine.get_parsing(p_img)
    parsing = engine.postprocess_parsing(parsing,*crop_img.shape[:2])
    import matplotlib.pyplot as plt
    plt.matshow(parsing)
    plt.colorbar()
    plt.savefig('parsing.png')
    # get id feature
    id_inp = engine.preprocess_faceid(align_face)
    id_feature = engine.get_id(id_inp)

    
    # get gender 
    gender_int,_ = engine.preprocess_gender(align_face)
    gender_out = engine.get_gender(gender_int)
    gender_out = engine.postprocess_gender(gender_out)


    # get 3d coeff
    deep3d_int = engine.preprocess_deep3d(align_face)
    coeff = engine.get_deep3d_coeff(deep3d_int)
    lmk_map = engine.postprocess_deep3d(coeff,info['align_lmk'])
    
   