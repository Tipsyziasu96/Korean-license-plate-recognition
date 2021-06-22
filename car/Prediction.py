import cv2
import itertools
import os
import time
import numpy as np
from .Model import get_Model
from .parameter import letters
import argparse
from tensorflow.keras import backend as K

#딥러닝 속도 향상을 위한 부분
import os

def prediction():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #K.set_learning_phase(0)

    Alpa = {'a':'ㅁ', 'b':'ㅠ' ,'c':'ㅊ', 'd': 'ㅇ',
            'e': 'ㄷ','f': 'ㄹ','g':'ㅎ','h':'ㅎ',
            'i':'ㅑ','j':'ㅓ','k':'ㅏ','l':'ㅣ','n':'ㅜ',
            'm':'ㅡ','o':'ㅐ','p':'ㅔ','q':'ㅂ','r':'ㄱ',
            's':'ㄴ','t':'ㅅ','u':'ㅕ','v':'ㅍ','w':'ㅈ',
            'x':'ㅌ',"y":'ㅛ',"z":'ㅋ'}

    Region = {"A": "서울 ", "B": "경기 ", "C": "인천 ", "D": "강원 ", "E": "충남 ", "F": "대전 ",
            "G": "충북 ", "H": "부산 ", "I": "울산 ", "J": "대구 ", "K": "경북 ", "L": "경남 ",
            "M": "전남 ", "N": "광주 ", "O": "전북 ", "P": "제주 ", "1": "1", "3": "3", "4": "4"}

    Hangul = {"dk": "아", "dj": "어", "dh": "오", "dn": "우", "qk": "바", "qj": "버", "qh": "보", "qn": "부",
            "ek": "다", "ej": "더", "eh": "도", "en": "두", "rk": "가", "rj": "거", "rh": "고", "rn": "구",
            "wk": "자", "wj": "저", "wh": "조", "wn": "주", "ak": "마", "aj": "머", "ah": "모", "an": "무",
            "sk": "나", "sj": "너", "sh": "노", "sn": "누", "fk": "라", "fj": "러", "fh": "로", "fn": "루",
            "tk": "사", "tj": "서", "th": "소", "tn": "수", "gj": "허", "gk": "하", "gh": "호"}

    def decode_label(out): #디코드라벨
        # out : (1, 32, 42)
        #np.argmax는 가장 큰 원소 값의 인덱스반환
        out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
        #print("첫번째")
        #print(out_best)
        out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value #중복제거
        #print("두번째")
        #print(out_best)
        outstr = ''

        for i in out_best:
            if i < len(letters):
                outstr += letters[i]

        # print("아웃")
        # print(outstr)
        return outstr


   
    def label_to_hangul(label):  # eng -> hangul

        if len(label) == 9:
            flag = 0
            region = label[0]
            two_num = label[1:3]
            hangul = label[3:5]
            four_num = label[5:]
        else:
            flag = 1
            #region='Z'

            two_num = label[1:4]
            hangul = label[4:6]

            four_num = label[6:]
        if flag == 0:
            try:
                region = Region[region] if region != 'Z' else ''
            except:
                pass
            try:
                hangul = Hangul[hangul]
            except:
                pass
            return region + two_num + hangul + four_num
        else:
            try:
                hangul = Hangul[hangul]
            except:
                pass

            return two_num + hangul + four_num

    ####################################################################################################
    #argument 제거

    #parser = argparse.ArgumentParser()
       
    #parser.add_argument("-t", "--test_img", help="Test image directory", type = str, default="C:\\Users\\majic\\OneDrive\\바탕 화면\\capstone_final-master\\car_license_plate_recognition\\car\\test\\") #경로 설정
    #args = parser.parse_args()

    ####################################################################################################

    test_img = "media\\" #web을 통해 업로드한 사진 파일들은 media 폴더에 들어감으로 
    weight = "C:\\Users\\User\\Desktop\\workspace\\test3.hdf5" #폴더내에 위치시킨 weightfile의 dir

    # Get CRNN model
    model = get_Model(training=False)

    try:
        #model.load_weights(args.weight)
        model.load_weights(weight)
        print("...Previous weight data...")
    except:
        #pass
        raise Exception("No weight file!")


    # test_dir = args.test_img
    # test_imgs = os.listdir(args.test_img)
    test_dir = test_img #C:\\Users\\majic\\OneDrive\\바탕 화면\\capstone_final-master\\car_license_plate_recognition\\car\\test\\
    test_imgs = os.listdir(test_img)
    total = 0
    acc = 0
    letter_total = 0
    letter_acc = 0
    start = time.time()
    cnt = 0

    for test_img in test_imgs:
        acc_chk = 0
        img = cv2.imread(test_dir + test_img, cv2.IMREAD_GRAYSCALE)

        img_pred = img.astype(np.float32)
        img_pred = cv2.resize(img_pred, (128, 64))
        img_pred = (img_pred / 255.0) * 2.0 - 1.0
        img_pred = img_pred.T
        img_pred = np.expand_dims(img_pred, axis=-1)
        img_pred = np.expand_dims(img_pred, axis=0)

        net_out_value = model.predict(img_pred) 

        pred_texts = decode_label(net_out_value)

        if(len(test_imgs[cnt]) == 13 and test_imgs[cnt][0] != 'Z' and len(pred_texts)==10):
            pred_texts = pred_texts[1:]

    str = label_to_hangul(pred_texts)

    return str   #여기서 반환된 (한글로 convert된 str)이 views.py를 통해 화면에 출력됩니다.