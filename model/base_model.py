import onnxruntime as ort
import onnx 
import numpy as np

class Model:
    def __init__(self,model_path):
        self.lmk_mean =np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.lmk_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.lmk_image_size=[256,256]

        return ort.InferenceSession(
                            onnx.load(model_path).SerializeToString(),
                            providers=[ 
                            'CUDAExecutionProvider', 
                            'CPUExecutionProvider'])