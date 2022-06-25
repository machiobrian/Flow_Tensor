#draw rectangles over the faces to highlight them

import face_coordinates
from face_coordinates import faces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#display the image -> draw rectangles over the faces

def highlight_faces(image_path, faces):
    #display the image
    image = plt.imread(image_path)

    ax = plt.gca()

    #for each face, draw a rectangle over it based on coordinates
    for face in faces:
        x,y,width, height = face['box']
        face_boarder = Rectangle((x,y), width, height, fill=False, color='green')
        ax.add_patch(face_boarder)
    plt.show()

highlight_faces('iacocca_1.jpg', faces)