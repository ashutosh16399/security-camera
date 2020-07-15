# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:16:04 2020

@author: Ashut
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:24:44 2020

@author: Ashut
"""

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import face_recognition
import random, string
import time



# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video_capture = cv2.VideoCapture(0)
video_width = video_capture.get(3)
video_height = video_capture.get(4)

Ashu_image = face_recognition.load_image_file("C:/Users/Ashut/Desktop/projects/security cam/data/Ashu/15.jpg")
Ashu_face_encoding = face_recognition.face_encodings(Ashu_image)[0]

 # Load a second sample picture and learn how to recognize it.
Pranav_image = face_recognition.load_image_file("C:/Users/Ashut/Desktop/projects/security cam/data/Pranav/11.jpg")
Pranav_face_encoding = face_recognition.face_encodings(Pranav_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
        Ashu_face_encoding,
        Pranav_face_encoding
    ]
known_face_names = [
        "Ashu",
        "Pranav"
    ]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

flag = 0

now = time.time()
future = now 

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    b_frame = imutils.resize(frame, width=min(1000, frame.shape[1]))
    gray = cv2.cvtColor(b_frame, cv2.COLOR_BGR2GRAY)
    
    
    (rects, weights) = hog.detectMultiScale(gray, winStride=(6, 6),
		padding=(8, 8), scale=1.3)
    
    
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        if time.time() > future:
            print(1)
            flag = 1
            now = time.time()
            future = now + 4
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    name = "Unknown"

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        
    if flag & (time.time() > future-1) :
        x = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        print(x)
        flag = 0
        print(name)
        cv2.imwrite(x+'.png', frame)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()