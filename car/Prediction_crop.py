import cv2
import itertools
import os
import time
import numpy as np
from .Model import get_Model
from .parameter import letters
import argparse
from tensorflow.keras import backend as K

import cv2
import imutils
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import os 

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#딥러닝 속도 향상을 위한 부분
import os



def prediction_crop():

    file_list = os.listdir('C:\\Users\\User\\Desktop\\workspace\\LP-final\\car_license_plate_recognition\\media\\')
    print(file_list)

    if ('1.jpg' in file_list):
        img = cv2.imread('C:\\Users\\User\\Desktop\\workspace\\LP-final\\car_license_plate_recognition\\media\\1.jpg',cv2.IMREAD_COLOR)
        # os.remove('C:\\Users\\User\\Desktop\\workspace\\LP-final\\car_license_plate_recognition\\media\\1.jpg')

    if ('2.jpg' in file_list):
        img = cv2.imread('C:\\Users\\User\\Desktop\\workspace\\LP-final\\car_license_plate_recognition\\media\\2.jpg',cv2.IMREAD_COLOR)
        os.remove('C:\\Users\\User\\Desktop\\workspace\\LP-final\\car_license_plate_recognition\\media\\2.jpg')


    # img = cv2.imread('C:\\Users\\User\\Desktop\\workspace\\LP-final\\car_license_plate_recognition\\media\\test.jpg',cv2.IMREAD_COLOR)
    
    img = cv2.resize(img, (600,400) )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15) 

    edged = cv2.Canny(gray, 10, 200) 



    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]


    screenCnt = None

    for c in contours:
        
        peri = cv2.arcLength(c, True)
        print(peri)
        approx = cv2.approxPolyDP(c, 0.05* peri, True)
    
        if (len(approx) == 4):
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)


    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]



    text = pytesseract.image_to_string(Cropped, config='--psm 11')

    img = cv2.resize(img,(500,300))
    Cropped = cv2.resize(Cropped,(200,65))


    cv2.imwrite('C:\\Users\\User\\Desktop\\workspace\\LP-final\\car_license_plate_recognition\\static\\img\\document.jpg',Cropped)
    cv2.imwrite('C:\\Users\\User\\Desktop\\workspace\\LP-final\\car_license_plate_recognition\\media\\document.jpg',Cropped)


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

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-w", "--weight", help="weight file directory",
    #                     type=str, default="./weight/LSTM+BN5--30--0.000.hdf5") #경로 설정
    # parser.add_argument("-t", "--test_img", help="Test image directory",
    #                     type=str, default="C:\\Users\Geon Yeol\Documents\.github\capstone_DB\\detection_image\\") #경로 설정
    # args = parser.parse_args()

    ####################################################################################################

    test_img = "C:\\Users\\User\\Desktop\\workspace\\LP-final\\car_license_plate_recognition\\media"
    weight = "C:\\Users\\User\\Desktop\\workspace\\test3.hdf5"


   # test_img = "C:\\Users\\User\\Desktop\\workspace\\car_license_plate_recognition\\media"

    
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
    test_dir = test_img
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
        img = cv2.imread('C:\\Users\\User\\Desktop\\workspace\\LP-final\\car_license_plate_recognition\\media\\document.jpg',cv2.IMREAD_GRAYSCALE)


        img_pred = img.astype(np.float32)
        img_pred = cv2.resize(img_pred, (128, 64))
        img_pred = (img_pred / 255.0) * 2.0 - 1.0
        img_pred = img_pred.T
        img_pred = np.expand_dims(img_pred, axis=-1)
        img_pred = np.expand_dims(img_pred, axis=0)

        net_out_value = model.predict(img_pred)

        #print(net_out_value)
        pred_texts = decode_label(net_out_value)
        ##########################################
        #예측값
        print()
        # print("###############################예측값####################################")
        #print(pred_texts)
        print()

        for i in range(min(len(pred_texts), len(test_img[0:-4]))):
            if pred_texts[i] == test_img[i]:
                letter_acc += 1

        letter_total += max(len(pred_texts), len(test_img[0:-4]))

            # print(pred_texts)
            # print(test_imgs[cnt])
            # print(len(test_imgs[cnt]))
            #원래 코드
            # if pred_texts == test_img[0:-4]:
            #     acc += 1
            #     acc_chk = 1
        if(len(test_imgs[cnt]) == 13 and test_imgs[cnt][0] != 'Z' and len(pred_texts)==10):
                pred_texts = pred_texts[1:]

        print("###############################예측값####################################")
            #print(pred_texts)
            #print(test_imgs[cnt])
        if pred_texts == test_img[0:-4]:
            print("일치")
            acc += 1
            acc_chk = 1
        else:
                #print(pred_texts)
                #print(test_img[0:-4])
                #print(test_img)
            print("불일치")
        total += 1
        cnt += 1
            # print("테스트이미지")
            # print(test_img[0:-4]) #실제 파일명
            # print(test_imgs) #전체 파일명들이 저장된 리스트
        print()
        if acc_chk == 1:
            print("Predicted: %s  /  True: %s  /  일치(%d / %d)"
                % ((label_to_hangul(pred_texts), label_to_hangul(test_img[0:-4]), acc, total)))
        else:
            print("Predicted: %s  /  True: %s"
                % ((label_to_hangul(pred_texts), label_to_hangul(test_img[0:-4]))))
        print()
            #이미지 출력확인문
            #cv2.rectangle(img, (0,0), (150, 30), (0,0,0), -1)
            #cv2.putText(img, pred_texts, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2) #원
            #cv2.putText(img, label_to_hangul(pred_texts), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

            #cv2.imshow("q", img)
            #if cv2.waitKey(0) == 27:
            #   break
            #cv2.destroyAllWindows()

    end = time.time()
    total_time = (end - start)

    print("적중률 : %f  (%d  /  %d)" % (acc / total, acc, total))
    print("Time : ", total_time / total)
    print("letter ACC : ", letter_acc / letter_total)

    str = label_to_hangul(pred_texts)

    return str
