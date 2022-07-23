'''
@author: LeslieZhao
@Date: 20220712
'''
import numpy as np
import math
import copy
from PIL import Image 
import scipy.spatial as spatial
import cv2
import math
from itertools import product as product


def nms(dets, thresh, opencv_mode=True):
    if opencv_mode:
        _boxes = dets[:, :4].copy()
        scores = dets[:, -1]
        _boxes[:, 2] = _boxes[:, 2] - _boxes[:, 0]
        _boxes[:, 3] = _boxes[:, 3] - _boxes[:, 1]
        keep = cv2.dnn.NMSBoxes(
            bboxes=_boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=0.,
            nms_threshold=thresh,
            eta=1,
            top_k=5000
        )
        if len(keep) > 0:
            return keep.flatten()
        else:
            return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def box_decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    """ 
    boxes = loc.copy()
    boxes[:, 0:2] = priors[:, 0:2] + boxes[:, 0:2] * variances[0] * priors[:, 2:4]       
    boxes[:, 2:4] = priors[:, 2:4] * np.exp(boxes[:, 2:4] * variances[1])

    # (cx, cy, w, h) -> (x, y, w, h)
    boxes[:, 0:2] -= boxes[:, 2:4] / 2

    # xywh -> xyXY
    boxes[:, 2:4] += boxes[:, 0:2]       
    # landmarks
    if loc.shape[-1] > 4:
        boxes[:, 4::2] = priors[:, None, 0] + boxes[:, 4::2] * variances[0] * priors[:, None, 2]
        boxes[:, 5::2] = priors[:, None, 1] + boxes[:, 5::2] * variances[0] * priors[:, None, 3]
    return boxes

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def anchor_fn(shape):
    min_sizes_cfg = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    steps = [8, 16, 32, 64]
    ratio = [1.]
    clip = False

    feature_map_2th = [int(int((shape[0] + 1) / 2) / 2),
                    int(int((shape[1] + 1) / 2) / 2)]
    feature_map_3th = [int(feature_map_2th[0] / 2),
                            int(feature_map_2th[1] / 2)]
    feature_map_4th = [int(feature_map_3th[0] / 2),
                            int(feature_map_3th[1] / 2)]
    feature_map_5th = [int(feature_map_4th[0] / 2),
                            int(feature_map_4th[1] / 2)]
    feature_map_6th = [int(feature_map_5th[0] / 2),
                            int(feature_map_5th[1] / 2)]

    feature_maps = [feature_map_3th, feature_map_4th,
                            feature_map_5th, feature_map_6th]
    anchors = []
    for k, f in enumerate(feature_maps):
        min_sizes = min_sizes_cfg[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                cx = (j + 0.5) * steps[k] / shape[1]
                cy = (i + 0.5) * steps[k] / shape[0]
                for r in ratio:
                    s_ky = min_size / shape[0]
                    s_kx = r * min_size / shape[1]
                    anchors += [cx, cy, s_kx, s_ky]
    # back to torch land
    output = np.array(anchors).reshape(-1, 4)
    if clip:
        output.clip(max=1, min=0)
    return output

def select_landmark(face_landmarks):
    assert len(face_landmarks) == 68
    lm_eye_left      = face_landmarks[36 : 42]  # left-clockwise
    lm_eye_right     = face_landmarks[42 : 48]  # left-clockwise
    lm_mouth_outer   = face_landmarks[48 : 60]  # left-clockwise
    lm_mouth_inner   = face_landmarks[60 : 68]  # left-clockwise

    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    nose         = face_landmarks[30]

    landmarks = np.zeros([5,2])
    landmarks[0,0] = eye_left[0]
    landmarks[0,1] = eye_left[1]
    landmarks[1,0] = eye_right[0]
    landmarks[1,1] = eye_right[1]
    landmarks[2,0] = nose[0]
    landmarks[2,1] = nose[1]
    landmarks[3,0] = mouth_left[0]
    landmarks[3,1] = mouth_left[1]
    landmarks[4,0] = mouth_right[0]
    landmarks[4,1] = mouth_right[1]
    return landmarks
    
    
def crop_image(image, boxes,s=1.15,size=(256,256)):
    x1, y1, x2, y2 = (boxes[:4] + 0.5).astype(np.int32)
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    center = np.array([cx, cy])

    scale = max(math.ceil(x2) - math.floor(x1),
                math.ceil(y2) - math.floor(y1)) / 200.0
    
    height = scale * 200.0 * s 

    top = (center - np.array([height / 2.0, height / 2.0])).astype(np.int32)
    bottom = (center + np.array([height / 2.0, height / 2.0])).astype(np.int32)

    crop_img = image[top[1]:bottom[1],top[0]:bottom[0]]
    
    if size is not None:
        crop_img = cv2.resize(crop_img,size)
        

    
    return crop_img,top,height


   
def preprocess(input_image_np,size,mean,std):
    input_image_np = cv2.resize(input_image_np, size)
    input_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2RGB)
    input_image_np = input_image_np.transpose((2,0,1)) / 255.
    input_image_np = np.array(input_image_np[np.newaxis, :])
    if mean is not None:
        input_image_np[:,0,...] = (input_image_np[:,0,...] - mean[0]) / std[0]
        input_image_np[:,1,...] = (input_image_np[:,1,...] - mean[1]) / std[1]
        input_image_np[:,2,...] = (input_image_np[:,2,...] - mean[2]) / std[2]
    return input_image_np

    

def get_preds(scores):
    """
    get predictions from score maps in numpy array
    """
    b,n,h,w = scores.shape
    idx = np.argmax(scores.reshape(b,n, h*w), axis = 2)
    idx = idx + 1

    preds = np.zeros([b,n,2])
    preds[:, :, 0] = (idx - 1) % w + 1
    preds[:, :, 1] = np.floor((idx - 1) / w) + 1

    for bi in range(b):
        for ni in range(n):
            hm = scores[bi][ni]
            px = int(math.floor(preds[bi][ni][0]))
            py = int(math.floor(preds[bi][ni][1]))
            if (px > 1) and (px < w) and (py > 1) and (py < h):
                preds[bi][ni][0] += np.sign(hm[py - 1][px] - hm[py - 1][px - 2]) * .25
                preds[bi][ni][1] += np.sign(hm[py][px - 1] - hm[py - 2][px - 1]) * .25
    preds += 0.5
    return preds

def convert_lmk_map(pred_face_proj,lmk):
    
    mix_face_proj = pred_face_proj
    mix_face_proj[:,0] = mix_face_proj[:,0]/224*256
    mix_face_proj[:,1] = 256 - mix_face_proj[:,1]/224*256
    tree = spatial.KDTree(mix_face_proj[:, :2])
    dis, idx = tree.query(lmk, k=5)
    landmark_mapping = np.zeros(17, dtype=int)
    for i in range(17):
        depth = mix_face_proj[idx[i], 2]
        pick = np.argmin(depth)
        landmark_mapping[i] = idx[i, pick]
    return landmark_mapping







def get_rotate_lmks(RotateMatrix, lmks):
    rotated_lmks = copy.copy(lmks)
    for i in range(len(lmks)):
        rotated_lmks[i, 0] = RotateMatrix[0][0] * lmks[i, 0] + RotateMatrix[0][1] * lmks[i, 1] + RotateMatrix[0][2]
        rotated_lmks[i, 1] = RotateMatrix[1][0] * lmks[i, 0] + RotateMatrix[1][1] * lmks[i, 1] + RotateMatrix[1][2]
    return rotated_lmks



def align_crop_img(input_img, lmks, std_mask=None,size=256):
    
    img_h,img_w = input_img.shape[0],input_img.shape[1]
    img_box = [np.min(lmks[:, 0]), np.min(lmks[:, 1]), np.max(lmks[:, 0]), np.max(lmks[:, 1])]
    
    center = ((img_box[0] + img_box[2]) / 2.0, (img_box[1] + img_box[3]) / 2.0)
    angle = np.arctan2((lmks[97, 1] - lmks[96, 1]), (lmks[97, 0] - lmks[96, 0])) / np.pi * 180

    RotateMatrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_lmks = get_rotate_lmks(RotateMatrix, lmks)
    faceBox = [np.min(rotated_lmks[:, 0]), np.min(rotated_lmks[:, 1]),
            np.max(rotated_lmks[:, 0]), np.max(rotated_lmks[:, 1])]

    cx_box = (faceBox[0] + faceBox[2]) / 2.
    cy_box = (faceBox[1] + faceBox[3]) / 2.
    width = faceBox[2] - faceBox[0] + 1
    height = faceBox[3] - faceBox[1] + 1
    face_size = max(width, height)
    bbox_size = face_size
    shift_x = 0
    shift_y = 0

    expand_size = bbox_size *0.35
    cropimg_size_x = int(round(bbox_size + expand_size))
    cropimg_size_y = int(round(bbox_size + expand_size * 3))
    x_min = int(round((cx_box - shift_x) - bbox_size / 2. - expand_size*0.5))
    y_min = int(round((cy_box - shift_y) - bbox_size / 2. - expand_size*2))
    x_max = x_min + cropimg_size_x
    y_max = y_min + cropimg_size_y
    boundingBox = [x_min, y_min, x_max, y_max]
    b_x_min,b_y_min,b_x_max,b_y_max = boundingBox

    # 获取旋转正人脸在的box在原图位置的外接矩形
    box_lmks = np.asarray([[b_x_min,b_y_min],[b_x_min,b_y_max],[b_x_max,b_y_max],[b_x_max,b_y_min]])
    M_r = cv2.invertAffineTransform(RotateMatrix)
    ori_box = get_rotate_lmks(M_r,box_lmks)

    outer_x_min = np.min(ori_box[:,0])
    outer_x_max = np.max(ori_box[:,0])
    outer_y_min = np.min(ori_box[:,1])
    outer_y_max = np.max(ori_box[:,1])
    
    ori_shift = [0,0,0,0]
    outer_x_min_tmp = outer_x_min
    outer_x_max_tmp = outer_x_max 
    outer_y_min_tmp = outer_y_min
    outer_y_max_tmp = outer_y_max

    outer_y_max_bk = outer_y_max
    outer_x_max_bk = outer_x_max
    
    img = input_img.copy()

    if outer_x_min < 0:
        img = cv2.copyMakeBorder(img,0,0,-outer_x_min,0,cv2.BORDER_REPLICATE  )
        ori_shift[0] = -outer_x_min
        outer_x_max_tmp = outer_x_max - outer_x_min
        outer_x_min_tmp = 0
        
    if outer_y_min < 0:
        img = cv2.copyMakeBorder(img,-outer_y_min,0,0,0,cv2.BORDER_REPLICATE  )
        ori_shift[1] = -outer_y_min
        outer_y_max_tmp = outer_y_max - outer_y_min
        outer_y_min_tmp= 0

    if outer_x_max > img_w:
        img = cv2.copyMakeBorder(img,0,0,0,outer_x_max - img_w,cv2.BORDER_REPLICATE  )
        ori_shift[2] = outer_x_max - img_w

    if outer_y_max > img_h:
        img = cv2.copyMakeBorder(img,0,outer_y_max - img_h,0,0,cv2.BORDER_REPLICATE  )
        ori_shift[3] = outer_y_max - img_h

    # pdb.set_trace()
    outer_x_min = outer_x_min_tmp
    outer_x_max = outer_x_max_tmp 
    outer_y_min = outer_y_min_tmp
    outer_y_max = outer_y_max_tmp

    align_lmks = lmks + np.array([ori_shift[0],ori_shift[1]])
    align_lmks = align_lmks - np.array([outer_x_min,outer_y_min])

    outer_img = img[outer_y_min:outer_y_max,outer_x_min:outer_x_max]
    outer_h, outer_w = outer_y_max - outer_y_min, outer_x_max - outer_x_min
   
    new_RotateMatrix = cv2.getRotationMatrix2D((float((outer_x_max - outer_x_min)//2),float((outer_y_max-outer_y_min)//2)), angle, scale=1)
    
    rotated_img = cv2.warpAffine(outer_img, new_RotateMatrix, (outer_img.shape[1], outer_img.shape[0]),borderMode=cv2.BORDER_REPLICATE  )
    align_lmks = get_rotate_lmks(new_RotateMatrix,align_lmks)

    # pdb.set_trace()
    
    rotate_h, rotate_w = b_y_max - b_y_min, b_x_max - b_x_min
    offset_diff = [outer_h - rotate_h, outer_w - rotate_w]
    offset_box = [offset_diff[0]//2,offset_diff[1]//2,offset_diff[0]-offset_diff[0]//2,offset_diff[1]-offset_diff[1]//2]
    #import pdb;pdb.set_trace()
    imgCropped = rotated_img[offset_box[1]:outer_h-offset_box[3],offset_box[0]:outer_w-offset_box[2]]
    crop_size = [imgCropped.shape[1],imgCropped.shape[0]]
    align_lmks = align_lmks - np.array([offset_box[0],offset_box[1]])

    
    scale = max(256/crop_size[0],256/crop_size[1])

    fix_face_box = [int(crop_size[0] * scale), int(crop_size[1] * scale)] 
    fix_outer_box = [int(outer_w * scale), int(outer_h * scale)]

    fix_new_RotateMatrix = cv2.getRotationMatrix2D(((outer_x_max - outer_x_min)*scale//2,(outer_y_max-outer_y_min)*scale//2), angle,scale=1 )

    # resize原始图加速ronghe
    outer_img_r = cv2.resize(outer_img,(fix_outer_box[0],fix_outer_box[1]))
    rotated_img_r = cv2.resize(rotated_img,(fix_outer_box[0],fix_outer_box[1]))
    
    imgResize = cv2.resize(imgCropped, (size, size))
    align_lmks = align_lmks / np.array(crop_size) * np.array([size,size])

    new_RotateMatrix_r = cv2.invertAffineTransform(fix_new_RotateMatrix)

    fix_ori_box = [int(ori_shift[0]*scale),
                        int(ori_shift[1]*scale),
                        int(ori_shift[2]*scale),
                        int(ori_shift[3]*scale)]

    fix_offset_box = [int(offset_box[0]*scale),
                        int(offset_box[1]*scale),
                        int(offset_box[0]*scale + fix_face_box[0]),
                        int(offset_box[1]*scale + fix_face_box[1])]
    # mask 处理
    
    if std_mask is not None:    
        mask_crop = std_mask
        mask_crop_r = cv2.resize(mask_crop,(fix_face_box[0],fix_face_box[1])) 
        mask = np.zeros_like(outer_img_r)
        
        mask[fix_offset_box[1]:fix_offset_box[3],fix_offset_box[0]:fix_offset_box[2]] = mask_crop_r

        fix_mask = np.zeros_like(outer_img_r)
        fix_mask[fix_ori_box[1]:outer_img_r.shape[0]-fix_ori_box[3],fix_ori_box[0]:outer_img_r.shape[1]-fix_ori_box[2]] = 1.0
        fix_mask = cv2.warpAffine(fix_mask,fix_new_RotateMatrix, (outer_img_r.shape[1], outer_img_r.shape[0]))
        fix_mask = cv2.warpAffine(fix_mask, new_RotateMatrix_r, (outer_img_r.shape[1], outer_img_r.shape[0]))

        mask =  cv2.warpAffine(mask, new_RotateMatrix_r, (outer_img_r.shape[1], outer_img_r.shape[0]))  * fix_mask
    else:
        mask = None

    info = {'M':new_RotateMatrix_r}
    info['align_lmk'] = align_lmks
    info['face'] = imgResize
    info['outer_r'] = outer_img_r
    info['rotate_img_r'] = rotated_img_r
    info['fix_offset_box'] = fix_offset_box
    info['fix_face_box'] = fix_face_box
    info['fix_new_RotateMatrix'] = fix_new_RotateMatrix
    
    info['ori_size'] = [outer_w,outer_h]
    info['outer_box'] = [outer_y_min,min(outer_y_max_bk,img_h),outer_x_min,min(outer_x_max_bk,img_w)]
    info['mask'] = mask

    info['fix_outer_box'] = fix_outer_box 
    
    info['ori_offset'] = ori_shift
    
    return info


def draw_lmk(img,lmk):
    draw_img = img.copy()
    for p in lmk:
        cv2.circle(draw_img,(int(p[0]),int(p[1])),10,[0,255,0])
    return draw_img 