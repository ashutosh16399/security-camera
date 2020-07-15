# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:09:19 2020

@author: Ashut
"""
import os
import face_recognition
import cv2
import numpy as np


video_capture = cv2.VideoCapture(0)
main_dic={}
enc_list=[]

data_path ='C:/Users/Ashut/Desktop/projects/security cam/data'
data_dir_list = os.listdir(data_path)

for images in data_dir_list:
    img_list=os.listdir(data_path+'/'+ images)
    for img in img_list:
        tim = face_recognition.load_image_file(data_path + '/' + images + '/' + img)
        tim_encoding = face_recognition.face_encodings(tim,model="large")
        enc_list.append(tim_encoding)
    main_dic[images] = enc_list
    
value=[]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations , model="large")

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            name = "Unknown"
            for key,value in main_dic.items():
                
                for j in value:
                    matches = face_recognition.compare_faces(j, face_encoding,tolerance=0.5)
                    if True in matches:
                        break
                if True in matches:
                    name = key
                
            face_names.append(name)
            
    process_this_frame = not process_this_frame
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
    
