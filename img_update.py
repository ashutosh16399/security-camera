# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:50:42 2020

@author: goura
"""

import pyrebase

config={
   "apiKey": "AIzaSyBvg28jbnyDGDPz-JDJIEzG8Z7CpdPXF5I",
  "authDomain": "surveillance-fa914.firebaseapp.com",
  "databaseURL": "https://surveillance-fa914.firebaseio.com",
  "projectId": "surveillance-fa914",
  "storageBucket": "surveillance-fa914.appspot.com",
  "messagingSenderId": "201532838368",
  "appId": "1:201532838368:web:848d5e36fe83922ab75b85",
  "measurementId": "G-TDHEJMJXMM"
}

firebase=pyrebase.initialize_app(config)
db=firebase.database()
str=firebase.storage()
#unique id
'''name the file u want to upload as x.jpg where x is the generated key'''

#we upload the file here
str.child("abce"+".jpg").put("abce"+".jpg")

#set your values to data here
data={"id":"abce","time":"7:00pm","name":"cheetah"}
db.child("img_list").push(data)