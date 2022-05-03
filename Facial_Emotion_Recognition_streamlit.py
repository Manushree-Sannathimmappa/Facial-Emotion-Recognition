# This is the code to run as application
# You will be asked to turn the camera on for recognizing the facial emotions.

from tensorflow import keras
import numpy as np
import streamlit as st
import cv2


st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .main{
    background-color: #FFFFF0;
        }
    
    </style>
    """,
    unsafe_allow_html = True  
    )

model = keras.models.load_model('saved_model_VGG_2_10 epochs.h5')

st.title("Facial Emotion Recognition")
st.subheader("Please check the below box to open up the web camera for facial emotion recognition")


on = st.checkbox("Turn Camera on")


output_image = st.image([])
cap = cv2.VideoCapture(0)

if on:
    cv2.ocl.setUseOpenCL(False)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    while cap.isOpened():
        ret, frame = cap.read() 
        
        if not ret:
            break
        bounding_box = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50),(x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray_frame, (48, 48))
            cropped_img = cropped_img.reshape(1,48,48,3)
            emotion_prediction = model.predict(cropped_img/255)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        frame = cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output_image.image(frame)
        
else:
    cap.release()
    




