#%%
from batch_dataset import DataLoader
from feature_extractor import FeatureExtractor
from roi_pooling import RoIPooling
from mini_batch import mini_batch
from detectron import Detectron


input_dir = "/home/maciek/Documents/images/schematic/img/"
target_dir = "/home/maciek/Documents/images/schematic/ann/"

dataset= DataLoader(input_dir, target_dir)
img=[i[0] for i in dataset]
bbox=[i[1][0] for i in dataset]
cls=[i[1][1] for i in dataset]
#%%
features=FeatureExtractor(l2=0.01)
features=[features.call(j) for j in img]
all_dataset=RoIPooling()
all_dataset=all_dataset.all_roi(features, cls, bbox)
#%%
roi_list=[]
cls_list=[]
bbox_list=[]
for k in all_dataset:
     roi_list.append(k[0])
     cls_list.append(k[1])
     bbox_list.append(k[2])

#%%
mini=mini_batch(roi_list, bbox_list, cls_list)
# %%
detectron=Detectron()
cls_pred=[detectron.call(l[1])[0] for l in mini]
cls_loss=[detectron.cls_results(m[2], n) for m,n in zip(mini, cls_pred)]
reg_pred=[detectron.call(o[1])[1] for o in mini]
reg_loss=[detectron.reg_results(p[2], p[3], r) for p,r in zip(mini,reg_pred)]
print(len(cls_loss))
print(len(reg_loss))
# %%
