import cv2
import numpy as np
import argparse
import os
import sys

from .lpDetector import *



#딥러닝 속도 향상을 위한 부분
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(__file__) + '/'

PATH_TO_CKPT = 'model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'protos/lp_label_map.pbtxt'




class lpDetector:
    def lp_detection(self):
        #lp_detector = lpDetector()



        #######################################################
        #img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
        #cv2.imshow('image', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #######################################################


        #ap = argparse.ArgumentParser()
        #ap.add_argument('image_file', help='image_file_to_run_inference',
        #                value=img)
        #ap.add_argument('C:\\Users\Geon Yeol\Desktop\\20-1\capstone\capstone_github\License-Plate-Detection-for-Embedded-Systems-master\\test.jpg', help='image_file_to_run_inference')
        #args = ap.parse_args()

        print("처음")
        #frame = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
        path = "C:\\Users\\Jun_PC\\anaconda3\\envs\\mysite\\media\\123.jpg"
        #All_img = os.listdir(path)
        #for test_img in All_img:
        #    print(path + test_img)
        # img = "Z54rk0639.jpg" #사진명
        frame = cv2.imread(path, cv2.IMREAD_COLOR)

        #출력
        height, width = frame.shape[:2]
        #print("image file:", args.image_file, "(%dx%d)" % (width, height))

        frame = frame[:, :, 0:3]
        (boxes, scores, classes) = self.detect(frame)
        vl_boxes = boxes[np.argwhere(scores > 0.3).reshape(-1)]
        vl_scores = scores[np.argwhere(scores > 0.3).reshape(-1)]

        if len(vl_boxes) > 0: #디텍팅 성공
            for i in range(len(vl_boxes)):
                box = vl_boxes[i]
                cropped_vl = frame[box[0]:box[2], box[1]:box[3], :]
                print(box)
                path = "C:\\Users\\Jun_PC\\anaconda3\\envs\\mysite\\media\\"
                os.chdir(path)  # 저장 경로 변경
                cv2.imwrite('test_123', cropped_vl) #저장할 파일명
                cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (200, 255, 0), 2)
        else:
            print('Unable to align')
            sys.exit() #반복문일 때 수정할 것

        print("press any key to quit")

        cv2.imshow("Frame", frame)

        #cropped한 이미지 확대 출력
        cropped_vl = cv2.resize(cropped_vl, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("cropped", cropped_vl)

        #CRNN.Prediction() #예측값 실행

        #CRNN 연동
        #import CRNN.Prediction

        #CRNN 연동
        print("예측값 실행")
        from .Prediction import prediction #이부분을 맨 위로 올리면 실행순서가 역순됨

        str = prediction()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return str

    