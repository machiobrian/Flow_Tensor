import cv2 as cv
from mtcnn.mtcnn import MTCNN

#to stream a video, we have to create an a video capture object with 
# the device
capture_stream = cv.VideoCapture(0) #0 -> webcam

"""Capture state remains continous until interrupted"""
while True:
    _, frame = capture_stream.read()

    """"Create an instance of the MTCNN function/class using the default weights"""
    detector = MTCNN()
    #this internal method, detects faces in the video strream
    result = detector.detect_faces(frame)
    print(result)
    #inspect the emptiness of the face 
    if result != []:
        """if result is not 0, trace a rectangle around with facial keypoints"""
        for person in result:
            traced_rect = person['box']
            key_points = person['keypoints']
    
        """Trace Rectangle enclosing the face"""
        cv.rectangle(frame, (traced_rect[0], traced_rect[1]),
                            (traced_rect[0] + traced_rect[2],
                            traced_rect[1] + traced_rect[3]),
                            (0,155,255),
                             1)
    


    """Show tracing in the video stream"""
    cv.imshow('frame', frame)

    #stop stresm upon q press
    if cv.waitKey(1) & 0xff == ord('q'):
        break