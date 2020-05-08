import imageio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

global TOTAL_ITERS
TOTAL_ITERS = 34000

import tensorflow.compat.v1.keras.backend as K
from detector.face_detector import MTCNNFaceDetector
import glob

from preprocess import preprocess_video
K.clear_session()

fn_source_video = "source.mov"
fn_target_video = "target.mp4"

fd = MTCNNFaceDetector(sess=K.get_session(), model_path="./mtcnn_weights/")

save_interval = 5 # perform face detection every {save_interval} frames
save_path = "./faceA/"
preprocess_video(fn_source_video, fd, save_interval, save_path)
save_path = "./faceB/"
preprocess_video(fn_target_video, fd, save_interval, save_path)

print(str(len(glob.glob("faceA/rgb/*.*"))) + " face(s) extracted from source video: " + fn_source_video + ".")
print(str(len(glob.glob("faceB/rgb/*.*"))) + " face(s) extracted from target video: " + fn_target_video + ".")

from keras.layers import *
import keras.backend as K
import tensorflow as tf