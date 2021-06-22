from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Image_Generator import TextImageGenerator
from Model import get_Model
from parameter import *

#딥러닝 속도 향상을 위한 부분
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#K.set_learning_phase(0) 원래있던 것이나, 새로운 keras에서는 버그가 수정되어 필요없음

# # Model description and training

model = get_Model(training=True)

try:
    model.load_weights('./weight/weight_0.961.hdf5') #이미 학습하던 가중치데이터
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

#train_file_path = './DB/train/' #원래
train_file_path = 'C:\\Users\\Jun_PC\\anaconda3\\envs\\mysite\\car\\CRNN\\capstone_DB\\labeling\\'
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)
tiger_train.build_data()

#valid_file_path = './DB/test/' #원래
valid_file_path = 'C:\\Users\\Jun_PC\\anaconda3\\envs\\mysite\\car\\CRNN\\capstone_DB\\test\\'

tiger_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size, downsample_factor)
tiger_val.build_data()

ada = Adadelta()

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='./weight/LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

#출력확인
#print(int(tiger_train.n / batch_size))
#print(int(tiger_val.n / val_batch_size))

# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size), #원래
                    #steps_per_epoch=int(batch_size / tiger_train.n),
                    epochs=1, #원 30
                    callbacks=[checkpoint],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(tiger_val.n / val_batch_size)), #원래
                    #validation_steps=int(val_batch_size / tiger_val.n))
