import tensorflow as tf
import numpy as np
from mini_batch import IoU

class RoIPooling(tf.keras.layers.Layer):
    
    def __init__(self):
        super().__init__()
    
    def extract_roi(self, cls, bbox):
        a=0
        for i in cls:
            if i!=0:
                break     
            a+=1
        x=bbox[a]
        return x

    def pool_roi(self,roi, features):

        feature_map=tf.squeeze(features)
        feature_map_height = int(feature_map.shape[0])
        feature_map_width  = int(feature_map.shape[1])

        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width  * roi[1], 'int32')
        h_end   = tf.cast(feature_map_height * roi[2], 'int32')
        w_end   = tf.cast(feature_map_width  * roi[3], 'int32')

        region = feature_map[h_start:h_end, w_start:w_end, :]

        region_height = h_end - h_start
        region_width  = w_end - w_start
        
        h_step = tf.cast( region_height / 7, 'int32')
        w_step = tf.cast( region_width  / 7, 'int32')

        areas = [[(i*h_step, j*w_step, (i+1)*h_step if i+1 < 7 else region_height, (j+1)*w_step if j+1 < 7 else region_width) for j in range(7)] for i in range(7)]

        def pool_area(x):
                corners=list(x)
                if (corners[0]==corners[2])&(corners[1]==corners[3]):
                    if corners[2]<feature_map_height:
                        i=corners[2]
                        v=tf.cast(1, 'int32')
                        corners[2]=tf.add(i, v)
                    else:
                        i=corners[0]
                        v=tf.cast(-1, 'int32')
                        corners[0]=tf.add(i, v)
                    if corners[3]<feature_map_width:
                            i=corners[3]
                            v=tf.cast(1, 'int32')
                            corners[3]=tf.add(i, v)
                    else:
                            i=corners[1]
                            v=tf.cast(-1, 'int32')
                            corners[1]=tf.add(i, v)
                    x=corners
                    return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
                elif corners[0]==corners[2]&corners[1]!=corners[3]:
                    if corners[2]<feature_map_height:
                        i=corners[2]
                        v=tf.cast(1, 'int32')
                        corners[2]=tf.add(i, v)
                    else:
                        i=corners[0]
                        v=tf.cast(-1, 'int32')
                        corners[0]=tf.add(i, v)
                    x=corners
                    return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
                elif corners[1]==corners[3]&corners[0]!=corners[2]:
                    if corners[3]<feature_map_width:
                            i=corners[3]
                            v=tf.cast(1, 'int32')
                            corners[3]=tf.add(i, v)
                    else:
                            i=corners[1]
                            v=tf.cast(-1, 'int32')
                            corners[1]=tf.add(i, v)
                    x=corners
                    return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
                else:
                    x=corners
                    return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1]) 
        
        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])

        return pooled_features

    def multiple_roi(self, features, cls, bbox):
        for i,j in zip(cls, bbox):
            x=[]
            for a,b in zip(i,j):
                x.append(self.extract_roi(a, b))

            def helper(x):
                return self.pool_roi(x, features)

            pooled_areas=tf.stack([helper(i) for i in x])
            #print(x)
            #iou=tf.stack([IoU_tf(i) for i in x])
            #print(iou)
            #iou = tf.stack([d for a,d in zip(pooled_areas, iou) if np.max(a.numpy())>=0])
            filtered_roi=tf.stack([tf.cast(a, dtype=tf.float32) for a in pooled_areas if np.max(a.numpy())>=0])
            
            cls_filtered=tf.stack([b for a,b in zip(pooled_areas, i) if np.max(a.numpy())>=0])
            
            bbox_filtered=tf.stack([c for a,c in zip(pooled_areas, j) if np.max(a.numpy())>=0])

            return filtered_roi, cls_filtered, bbox_filtered
        
    def all_roi(self, features, cls, bbox):
        x=[features, cls, bbox]

        def helper2(x):
            return self.multiple_roi(x[0], x[1], x[2])
        
        all_dataset=helper2(x)
        
        return all_dataset