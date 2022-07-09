import pandas as pd
import os
import cv2 as cv
from datetime import datetime

global imgList, steeringList
countFolder = 0
count = 0
imgList = []
steeringList = []

#get the current directory path
myDirecory = os.path.join(os.getcwd(), 'DataCollected') #data collected is 
                                            #the name of the new folder
print(myDirecory)

#find the number of folders present in the data collected folder
while os.path.exists(os.path.join(myDirecory,f'IMG{str(countFolder)}')):
    countFolder += 1
newPath = myDirecory + "/IMG"+str(countFolder)
os.makedirs(newPath)

#save all the images in this new folder
def saveData(img, steering):
    global imgList, steeringList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '') #timestamps are unique, it cannot be replicated
    print('timestamp = ', timestamp)
    fileName = os.path.join(newPath,f'Image_{timestamp}.jpg')
    cv.imwrite(fileName, img)
    #store file name and the user angle in a list
    imgList.append(fileName) #has the file names not the images
    steeringList.append(steering)

#create a function to save the log files when the sessions end
def saveLog(): #transfers all the data into a csv file usinf pandas
    global imgList, steeringList
    rawData = {'Image': imgList,
                'Steering': steeringList}
    
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDirecory,f'log_{str(countFolder)}.csv'),
    index=False, header=False)
    print('Log Saved')
    print('Total Images: ', len(imgList))

#when the script is to run by itself, below is first run
#when the script is called, below is ignored
cap = cv.VideoCapture(0)
for x in range(10):
    _, img = cap.read()
    saveData(img, 0.5) #default steering value is 0.5, while the img is from the video stream
    cv.waitKey(1)
    cv.imshow('Image', img)
saveLog()