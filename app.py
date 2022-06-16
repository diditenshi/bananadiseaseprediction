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
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
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
     classifier_model = "vgg19-model.h5"
      
     model = load_model(classifier_model)
      
     test_image = image.resize((224,224))
     test_image = preprocessing.image.img_to_array(test_image)
     test_image = keras.applications.imagenet_utils.preprocess_input((test_image)
     test_image = np.expand_dims(test_image, axis=0)
     class_names = {0 : 'bunchy-top', 1 :'fusarium-wilt', 2 :'healthy', 3 :'moko', 4 :'sigatoka'}

     predictions = model.predict(test_image)
     scores = tf.nn.softmax(predictions[0])
     scores = scores.numpy()

    
     result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
     return result

if __name__ == "__main__":
    main()
