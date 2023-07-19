import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, TimeDistributed
import numpy as np
from keras import backend as K

from mini_batch import extract_roi

class Detectron(tf.keras.Model):
    
    def __init__(self):
        super().__init__()

        regularizer = tf.keras.regularizers.l2(0.01)
        class_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01)
        regressor_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.001)

        self.flatten=Flatten(name='flatten')
        self.fc1 = Dense(name='fc1',units=4096, activation='relu', kernel_regularizer=regularizer)
        self.dropout1=Dropout(name='drop1', rate=0.5)
        self.fc2 = Dense(name='fc2',units=4096, activation='relu', kernel_regularizer=regularizer)
        self.dropout2=Dropout(name='drop2', rate=0.5)

        self.cls_pred=Dense(name='cls_pred', units=9, activation='softmax', kernel_regularizer=class_initializer)
        
        self.reg_pred=Dense(name='reg_pred', units=32, activation='linear', kernel_regularizer=regressor_initializer)

    def call(self, roi_list):
        roi_list=tf.expand_dims(roi_list, axis=0)
        y=self.flatten(roi_list)
        fc1 = self.fc1(y)
        drop1=self.dropout1(fc1)
        fc2 = self.fc2(drop1)
        drop2=self.dropout2(fc2)

        cls_pred=self.cls_pred(drop2)
        
        reg_pred=self.reg_pred(drop2)

        return cls_pred, reg_pred
    
    def cls_results(self, cls_list, cls_pred):
        x=np.append([0], [cls_list])
        x=np.float32(x)
        x=tf.cast(x, dtype=tf.float32)
        a=cls_pred[0]
        #print(x)
        #print(a)
        cls_loss= K.sum(K.categorical_crossentropy(target=x, output=a, from_logits=False))
        return cls_loss

    def reg_results(self, cls_list, bbox_list, reg_pred):
        y_true, b =extract_roi(cls_list, bbox_list, True)
        y_pred=tf.cast([reg_pred[0][((b+1)*4)-4],reg_pred[0][((b+1)*4)-3],reg_pred[0][((b+1)*4)-2],reg_pred[0][((b+1)*4)-1]], dtype=tf.float32)
        z=y_true-y_pred
        z_abs=tf.math.abs(z)
        losses=[]
        for ind,r in enumerate(z_abs):
            if r<1.0:
                loss=0.5*z[ind]*z[ind]*1.0
                losses.append(loss)
            else:
                loss=r-0.5 / 1.0
                losses.append(loss)
        reg_losses=K.sum(tf.cast(losses, dtype=tf.float32))

        return reg_losses

