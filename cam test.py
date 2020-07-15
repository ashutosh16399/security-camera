# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:28:15 2020

@author: Ashut
"""
import os
import numpy as np
import cv2

filename = 'video.avi'
frames_per_second = 24.0
res = '480p'

cap=cv2.VideoCapture(0)

def change_res(cap,width, height):
    cap.set(3, width)
    cap.set(4, height)

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))


while True:
    _,frame = cap.read(0)
    
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('color testing cam',frame)
    
    out.write(frame)
    
    #cv2.imshow('black and white testing cam',gray)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows() 