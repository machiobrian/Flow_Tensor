
from unittest import result
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mtcnn.mtcnn import MTCNN

#file_name = 'image1.jpg'
plot_image = plt.imread('ConvNeu_Net/Multitask_Cascaded_CNN/image2.jpg')

#create an instance of the MCTNN() function/class, using
#default weights
detector = MTCNN()

#detect_face method to identify faces in an image
face_detection = detector.detect_faces(plot_image)

print(face_detection)

#----------------------------------#
#function to trace rectangle and landmark points

def trace_face(plot_image, result_list):
    plt.imshow(plot_image)
    axes = plt.gca() #create a plot to draw the boxes

    for result in result_list:
        """"get the face for every rectangle from result by
        MTCNN()"""
        x,y,width, height = result['box']

        """call the rectangular funciton using above coordinate
        values"""
        trace_rect = Rectangle((x,y), width, height, fill=False, 
        color='green')

        axes.add_patch(trace_rect) #traces rectangle encasing the faces

    # for key, value in result['keypoints'].items():
    #     # Call the circle function for coordinate values
    #     key_points = Circle(value, radius=2, color='red')# Trace the circular points on the faces
    #     axes.add_patch(key_points)

    plt.show()

trace_face(plot_image, face_detection)
