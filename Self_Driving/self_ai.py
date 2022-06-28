from time import time
import numpy as np #Used to process the image data
import cv2 as cv #used to capture image with the respective camera
from PIL import Image, ImageEnhance, ImageOps #used to process image data
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D #layers in the neural net
from keras.models import Sequential, model_from_json #json -> file saving from tf
from keras.optimizers import Adam
import tensorflow as tf

import blynklib
import RPi.GPIO as GPIO #raspi gpio lib

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM) # use physical pin numbering

motor1 = 13 #GPIO for motor1
motor2 = 12 #GPIO for motor2
GPIO.setup(motor1, GPIO.OUT)
GPIO.setup(motor2, GPIO.OUT)

"""Set Motors to Pwm, it changes based on our motor controller"""
motor1Servo = GPIO.PWM(motor1, 50)
motor1Servo.start(8)
motor2Servo = GPIO.PWM(motor2, 50)
motor2Servo.start(8)

#second change of duty cycle is due to motor-ctrller errors
motor1Servo.ChangeDutyCycle(7.5) 
motor2Servo.ChangeDutyCycle(7.5)

def servoControl(value):
    motor1Servo.ChangeDutyCycle(7.5 + value)
    motor2Servo.ChangeDutyCycle(7.5 - value)

class Agent:
    def __init__(self):
        self.userSteering = 0
        self.aiMode = False
        self.model = Sequential([
            Conv2D(32, (7,7), input_shape=(240,320,3),strides=(2,2), 
            activation='relu', padding='same'),
            MaxPool2D(pool_size=(5,5),strides=(2,2),padding='valid'),
            Conv2D(64,(4,4),(1,1),'same', activation='same'),
            MaxPool2D((4,4),strides=(2,2), padding='valid'),
            Conv2D(128,(4,4),strides=(1,1),activation='relu',padding='same'),
            MaxPool2D(pool_size=(5,5),strides=(3,3), padding=('valid')),
            Flatten(),

            Dense(384, activation='relu'),
            Dense(64, activation='relu',name='layer1'),
            Dense(8,activation='relu',name='layer2'),
            Dense(1,activation='linear',name='layer3')
        ])

        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.05))

        """used when importing the pretrained weights"""
        #self.model.load_weights("selfdrive.h5")

        print(self.model.summary())

        self.cap = cv.VideoCapture(0) #changes based on the input device
        #controls the cameras resolution
        self.cap(3,320)
        self.cap(4,240)

    def act(self, state): #the method for the AI behaving in autonomous mode
        state = np.reshape(state, (1,240,320,3))
        action = self.model.predict(state[0][0])
        action = (action * 2) - 1
        servoControl(action)
        return action

    """This is where the AI's Neural net Improves/ learns"""
    def learn(self, state, action):
        state = np.reshape(state, (1,240,320,3))
        history = self.model.fit(state, [action], batch_size=1,epochs=1,verbose=0)
        print("Loss: ", history.history.get("loss")[0])

    def getState(self):
        ret, frame = self.cap.read() #fetch the actual webcam image
        pic = np.array(frame) #convert into an array
        processedImg = np.reshape(pic, (240,320,3))/255 #reshapes the image
        return processedImg

    def observeAction(self):
        return (self.userSteering +1 )/2

agent = Agent()
BLYNK_AUTH = 'insert project code'
blynk = blynklib.Blynk(BLYNK_AUTH)

@blynk.handle_event('write v4') #we are using the pin v4 on the blynk app for steering 
def write_virtual_pin_handler(pin, value):
    print("value: ", float(value[0]))
    agent.userSteering = float(value[0]) #update AI memory of steering angle
    
    #change the motors to turn appropriately based on steering input
    servoControl(float(value[0])) 

@blynk.handle_event('write v2') #we use pin v2 on the app for autonomous drive/learning
def write_virtual_pin_handler(pin, value):
    agent.aiMode = False if value == 1 else True #changes the AI mode based on reading


counter = 0
while True:
    blynk.run()

    if agent.aiMode == False: #if the AI learning mode is off
        start  = start.time()
        state = agent.getState()
        action = agent.observeAction()
        counter += 1

        if counter % 1 == 0: #AI learns every iteration, can be changed for the AI not to learn every iteration
            start = time.time()
            agent.learn(state, action)
            agent.memory = []

        if counter % 50 == 0: #this how often AI changes its weights, can be altered
            agent.model.save_weights("selfdrive.h5") #save training data in this file
        print("framerate: ", 1/(time.time() - start))
    else:
        while agent.aiMode == True: #this is an autonomous loop
            start = time.time()
            state = agent.getState()
            action = agent.act(state)

            print("action, action")
            print("framerate: ", 1/(time.time() - start))