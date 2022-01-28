import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#from SegZoo
def standardize(img):
    #standardization using adjusted standard deviation

    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0/np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    del m, s, N
    #
    if np.ndim(img)==2:
        img = np.dstack((img,img,img))
    return img

#load model
filepath = './saved_model'
model = tf.keras.models.load_model(filepath, compile = True)
model.compile

#segmentation
def FRFsegment(input_img):

    img = standardize(input_img)
    img = np.expand_dims(img,axis=0)
    
    est_label = model.predict(img)
    
#     # Test Time AUgmentation   
#     est_label2 = np.flipud(model.predict((np.flipud(img)), batch_size=1))
#     est_label3 = np.fliplr(model.predict((np.fliplr(img)), batch_size=1))
#     est_label4 = np.flipud(np.fliplr(model.predict((np.flipud(np.fliplr(img))))))

#     #soft voting - sum the softmax scores to return the new TTA estimated softmax scores
#     pred = est_label + est_label2 + est_label3 + est_label4
    
    pred = est_label
    
    mask = np.argmax(np.squeeze(pred, axis=0),-1)
    
    #overlay plot
    p = plt.imshow(input_img,cmap='gray')
    p = plt.imshow(mask, alpha=0.4)
    p = plt.axis("off")
    return plt

examples = [['examples/nowc1.jpg'],['examples/FRF_c1_snap_2019090318.jpg']] 
FRFSegapp = gr.Interface(FRFsegment, gr.inputs.Image(shape=(512, 512)), "image", examples=examples).launch()
