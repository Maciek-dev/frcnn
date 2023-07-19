import tensorflow as tf
from keras import backend as K

from batch_dataset import DataLoader
from feature_extractor import FeatureExtractor
from roi_pooling import RoIPooling
from mini_batch import mini_batch
from detectron import Detectron
from tensorflow.python.framework import ops

class FastRCNN(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.features=FeatureExtractor(l2=0.01)
        self.all_dataset=RoIPooling()
        self.detectron=Detectron()
    
    def call(self, dataset):

        features=self.features.call(dataset[0])
        
        all_dataset=self.all_dataset.all_roi(features, dataset[1][1], dataset[1][0])

        mini=mini_batch(all_dataset[0], all_dataset[1], all_dataset[2])
        #print(mini)
        pred=[self.detectron.call(i[1]) for i in mini]

        for x,y in zip(pred, mini):
            cls_loss=self.detectron.cls_results(y[2],x[0])
            reg_loss=self.detectron.reg_results(y[2], y[3], x[1])
            loss=tf.reduce_sum(cls_loss)+tf.reduce_sum(reg_loss)
        return loss


input_dir = "/media/maciek/Data_linux/images/schematics/img"
target_dir = "/media/maciek/Data_linux/images/schematics/ann"

train=DataLoader(input_dir, target_dir, True)
val=DataLoader(input_dir, target_dir, False)

model=FastRCNN()

optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.01)

model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath='tmp/checkpoint',monitor='val_loss', mode='min', save_best_only=True)

#@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        loss=model.call(data)
        #ops.reset_default_graph()
    #print(loss)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
            #train_acc_metric.update_state(y, logits)
    #tf.reset_default_graph()
    return loss

#@tf.function
def test_step(data):
    val_loss = model(data, training=False)
    #tf.reset_default_graph()

    return val_loss

#model.summary()
epochs = 1
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    for step, data in enumerate(train):

        loss=train_step(data)
        #tf.reset_default_graph()

        if step % 10 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.2f"
                    % (step, float(loss))
                #    "Training loss (for one batch) at step %d"
                #    % (step)
                )
                print("Seen so far: %s samples" % ((step + 1)))

    for val_step, val_data in enumerate(val):

        val_loss=test_step(val_data)
        
        if val_step % 10 == 0:
                print(
                    "Validation loss (for one batch) at step %d: %.2f"
                    % (val_step, float(val_loss))
                )
                print("Seen so far: %s samples" % ((val_step + 1)))
