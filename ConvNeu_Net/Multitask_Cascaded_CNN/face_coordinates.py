import urllib3.request
from urllib.request import urlopen
import matplotlib.pyplot as plt #read images
from mtcnn.mtcnn import MTCNN # detect faces within an image

#temporarily store the image in our local file for analysis -> store_image function
def store_image(image_url, local_file_name):
    with urlopen(image_url) as resource:
        with open(local_file_name, 'wb') as f:
            f.write(resource.read())

#call the function with the URL and the local file that we want to store image
store_image('https://ichef.bbci.co.uk/news/320/cpsprodpb/5944/production/_107725822_55fd57ad-c509-4335-a7d2-bcc86e32be72.jpg',
            'iacocca_1.jpg')
store_image('https://www.gannett-cdn.com/presto/2019/07/03/PDTN/205798e7-9555-4245-99e1-fd300c50ce85-AP_080910055617.jpg?width=540&height=&fit=bounds&auto=webp',
            'iacocca_2.jpg')

image = plt.imread('iacocca_1.jpg')

#from MTCNN() use detect face method to detect faces in an image
detector = MTCNN()
faces = detector.detect_faces(image)
for face in faces:
    print(face)

#returns a box key containing the boundary of the faces within the image. 
#it has 4 values: 
    
    # 1. x,y coordinates of the top_left, vertex, width and height of the 
#rectangle containing the face.
    #2. confidence
    #3. keypoints -> the key conatins a dictionary conatining the features of a face 
# that were  detected alongside their cordinates.

#draw rectangles over the faces to highlight them
