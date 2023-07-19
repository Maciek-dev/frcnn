import tensorflow as tf
import numpy as np
import random


random.seed(42)

def extract_roi(cls, bbox, line=False):
        a=0
        #print(cls)
        for cl in cls:
            if cl!=0:
                break     
            a+=1
        x=bbox[a]
        if line==False:
            return x
        elif line==True:
            return x, a

def IoU(roi_list, bbox_list, cls_list):
    positive=[]
    negative=[]
    #print(cls_list.size())
    for m in range(len(cls_list.numpy())):

        loc=extract_roi(cls_list.numpy()[m], bbox_list.numpy()[m])
        a=[tf.cast(loc[0]*512, 'int32'), tf.cast(loc[1]*512, 'int32'),tf.cast(loc[2]*512, 'int32'),tf.cast(loc[3]*512, 'int32')]
        h_start = tf.cast(32 * loc[0], 'int32')*16
        w_start = tf.cast(32 * loc[1], 'int32')*16
        h_end   = tf.cast(32 * loc[2], 'int32')*16
        w_end   = tf.cast(32 * loc[3], 'int32')*16
        #print(loc, h_start, w_start, h_end, w_end)
        c1=np.maximum(a[0], h_start)
        c2=np.maximum(a[1], w_start)
        c3=np.minimum(a[2], h_end)
        c4=np.minimum(a[3], w_end)
        intersection=(c3-c1)*(c4-c2)
        f1=(a[3]-a[1])*(a[2]-a[0])
        f2=(w_end-w_start)*(h_end-h_start)
        union=f1+f2-intersection
        IoU=intersection/union
        if IoU>0.5:
            x=[IoU, roi_list[m], cls_list[m], bbox_list[m]]
            positive.append(x)
        elif 0.1<IoU<=0.5:
            x=[IoU, roi_list[m], cls_list[m], bbox_list[m]]
            negative.append(x)
    
    return positive, negative

def mini_batch(roi_list, bbox_list, cls_list):
    
    pos=[]
    neg=[]
    
    positive, negative = IoU(roi_list, cls_list, bbox_list)
    
    pos.append(positive)
    neg.append(negative)

    pos=sum(pos,[])
    neg=sum(neg,[])

    if len(pos)==0 and len(neg)>0:
        if len(neg)<64:
            mini_batch=neg[:len(neg)]+random.choices(neg, k=64-len(neg))
            return mini_batch
        else:  
            mini_batch=neg[:64]
            return mini_batch
    elif len(neg)==0 and len(pos)>0:
        if len(pos)<64:
            mini_batch=pos[:len(pos)]+random.choices(pos, k=64-len(pos))
            return mini_batch
        else:
            mini_batch=pos[:64]
            return mini_batch
    elif 0<len(pos)<16 and 0<len(neg)<48:
        pos=pos[:len(pos)]+random.choices(pos, k=16-len(pos))
        neg=neg[:len(neg)]+random.choices(neg, k=48-len(neg))
        mini_batch=pos+neg
        return mini_batch
    elif 0<len(pos)<16 and len(neg)>=48:
        pos=pos[:len(pos)]+random.choices(pos, k=16-len(pos))
        mini_batch=pos+neg[:48]
        return mini_batch
    elif len(pos)>=16 and 0<len(neg)<48:
        neg=neg[:len(neg)]+random.choices(neg, k=48-len(neg))
        mini_batch=pos[:16]+neg
        return mini_batch
    elif len(pos)>=16 and len(neg)>=48:
        mini_batch=pos[:16]+neg[:48]
        return mini_batch
