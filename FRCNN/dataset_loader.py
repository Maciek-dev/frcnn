import numpy as np
import json
import os
from operator import itemgetter
import tensorflow as tf
from sklearn.preprocessing import label_binarize

class DataLoader:
     
    def __init__(self, input_dir, target_dir):
          self.input_dir= input_dir
          self.target_dir=target_dir

    #input_dir = "/home/maciek/Documents/images/schematic/img/"
    #target_dir = "/home/maciek/Documents/images/schematic/ann/"
    def links(self):
        input_img_paths = sorted(
            [
                os.path.join(self.input_dir, fname)
                for fname in os.listdir(self.input_dir)
            ]
        )

        target_img_paths = sorted(
            [
                os.path.join(self.target_dir, fname)
                for fname in os.listdir(self.target_dir)
            ]
        )
        print(len(input_img_paths))
        '''
        TRAIN_LENGTH=len(input_img_paths)
        
        train_full_size=int(0.8*TRAIN_LENGTH)
        validation_full_size=int(0.2*TRAIN_LENGTH)

        img_train=input_img_paths[:train_full_size]
        targets_train=target_img_paths[:train_full_size]
        img_val=input_img_paths[train_full_size:]
        targets_val=target_img_paths[train_full_size:]
        '''
        return input_img_paths, target_img_paths


    def load_targets(self, target_img_paths):
        data=json.load(open(target_img_paths))
        def details_load(data):
            def details(data):        
                    class_names={"bg":0,"amacrine":1,"bipolar":2,"cone":3,"ganglion":4,"horizontal":5,"muller":6,"rod":7,"rpe":8}
                    size=[data['size']['height'],data['size']['width']]
                    outline=[]
                    class_idx=[]
                    for i in data['objects']:
                        x=(i['points']['exterior'])
                        class_idx.append(class_names[i['classTitle']])
                        line=[]
                        for j in range(len(x)):
                            line.append(tuple(x[j]))
                        outline.append(line)
                    return size, outline[1:], class_idx[1:]

            size, outline, class_idx = details(data)
            bbox=[]
            for j in range(len(outline)):
                x=outline[j]
                x_max=((max(x,key=itemgetter(0))[0])/size[1])
                y_max=((max(x,key=itemgetter(1))[1])/size[0])
                x_min=((min(x,key=itemgetter(0))[0])/size[1])
                y_min=((min(x,key=itemgetter(1))[1])/size[0])
                bbox_line=[x_min, y_min, x_max, y_max]
                bbox.append(bbox_line)
            
            return bbox, class_idx

        bbox_list=details_load(data)
        def details(bbox_list):
            x=label_binarize(bbox_list[1], classes=[1,2,3,4,5,6,7,8])
            class_idx=np.array(x, dtype=np.float32)
            bbox=[]
            for i in zip(bbox_list[0], bbox_list[1]):
                a=np.zeros([8,4], dtype=np.float32)
                y=np.full_like(a, -1)
                y[i[1]-1]=i[0]
                bbox_x=[np.array(y, dtype=np.float32)]
                bbox.append(bbox_x[0])
            bbox=np.expand_dims(bbox, axis=0)
            class_idx=np.expand_dims(class_idx, axis=0)
            return bbox, class_idx

        bbox, class_idx= details(bbox_list)
        return bbox, class_idx

    def load_image(self, input_img_paths):
        def image(img):
            image = tf.io.read_file(input_img_paths)
            def img_load(img):
                img = tf.image.decode_png(img, channels=3)
                img = tf.image.resize(img, (512,512))
                img = tf.cast(img, dtype=tf.float32) / 255.0
                img = tf.expand_dims(img, axis=0)
                return img

            img=img_load(image)
            return img

        img = image(input_img_paths)
        return img


input_dir = "/home/maciek/Documents/images/schematic/img/"
target_dir = "/home/maciek/Documents/images/schematic/ann/"
#print(input_dir)
dataset=DataLoader(input_dir, target_dir)
x,y=dataset.links()
print(len(x))