# -*- coding: utf-8 -*-
"""banana-disease.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/13CF_ITlv3WCmA00qUPpz2n7ysK0mo9AG
"""

# from google.colab import drive
# drive.mount('/content/drive')

import streamlit as st
from PIL import Image
# import matplotlib.pyplot as plt
# import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing
import time



## this is part of web app

## ----------------------------------------------- x -----------------------------------------x-------------------------x------------------##


# fig = plt.figure()

st.title('Banana Disease Classifier')

st.markdown("Prediction of the Banana Diseases: Bunchy Top, Moko, Sigatoka and Fusarium Wilt")

def main():
    file_uploaded = st.file_uploader("Select Image", type=["png","jpg","jpeg"])
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_btn = st.button("Classify")
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                # plt.imshow(image)
                # plt.axis("off")

                predictions = predict(image)

                time.sleep(1)
                st.success('Classified')
                st.write(predictions)



## This code is for saved model in format as H5 file


def predict(image):
     classifier_model = "classify-vgg19-model-final.h5"
      
     model = load_model(classifier_model)
      
     test_image = image.resize((224,224))
     test_image = preprocessing.image.img_to_array(test_image)
     test_image = test_image / 255.0
     test_image = np.expand_dims(test_image, axis=0)
     class_names = {0 : 'Healthy', 1 : 'Bunchy Top Disease', 2 : 'Fusarium Wilt Disease', 3 : 'Moko (Bacterial Wilt) Disease', 4 : 'Sigatoka Disease'}
     class_care_options = {0 : 'plant1', 1 : 'plant2', 2 : 'plant3', 3 : 'plant4', 4 : 'plant5'}

     predictions = model.predict(test_image)
     scores = tf.nn.softmax(predictions[0])
     scores = scores.numpy()

    
     result = f" The banana plant is infected with {class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence and must be prevented using the ff. options {class_care_options[np.argmax(scores)]}." 
     return result


## -----------------------------------------------------x---------------------------------------x--------------------------------------------##

if __name__ == "__main__":
    main()
