import numpy as np
import tensorflow as tf

#딥러닝 속도 향상을 위한 부분
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(__file__) + '/'

PATH_TO_CKPT = 'model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'protos/lp_label_map.pbtxt'


class lpDetector:
    def __init__(self):        
        self.detection_graph = tf.compat.v1.Graph()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph) #수정
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(BASE_DIR + PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.compat.v1.import_graph_def(od_graph_def, name='')

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')        
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')        
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def __del__(self):
        self.sess.close()

    def detect(self, image):
        
        image_expanded = np.expand_dims(image, axis=0)
        
        (boxes, scores, classes, num_detections) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        
        boxes[0, :, [0, 2]] = (boxes[0, :, [0, 2]]*image.shape[0])
        boxes[0, :, [1, 3]] = (boxes[0, :, [1, 3]]*image.shape[1])
        return np.squeeze(boxes).astype(int), np.squeeze(scores), classes