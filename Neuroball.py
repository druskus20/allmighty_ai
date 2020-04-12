from grabber import Grabber
import numpy as np
import cv2
import time
from scipy.linalg import norm

# This class will contain all the screen capture functionality
class Neuroball(object):
    grp = None
    dimmensions = None # Screen area

    attempt = 1
    
    buffer_write = None     # Write pointer
    buffer_read = None      # Read pointer

    attempt_startime = 0
    attempt_endtime = 0

    times = []              # List with the time progression of each attempt

    # sfrag example: (1, 26, 1601, 926) captures 1600x900 
    # without the window bar 
    def __init__(self, sfrag, buffer_len=2):
        self.grb = Grabber(bbox=sfrag)

        # Calculate screen size
        self.dimmensions = (sfrag[2]-sfrag[0], sfrag[3]-sfrag[1], 3)

        # Set the frame buffer to zeros
        self.buffer_write = np.zeros(self.dimmensions, dtype=np.int8)
        self.buffer_read = np.zeros(self.dimmensions, dtype=np.int8)


    def resize_image(self, image, res=(720, 576)):
        pr_image = cv2.resize(image, res)
        return np.asarray(pr_image, dtype=np.uint8)

    # Grabs a frame and sotores it in buffer[buffer_head]
    def refresh_frame(self):
        self.buffer_write = self.grb.grab(None)
        self.buffer_write, self.buffer_read = self.buffer_read, self.buffer_write
        
 

    # Gets the newest frame from buffer[buffer_head]
    def get_newest_frame(self):
        return self.buffer_read


    def recorder_thread(self, show=False):
        self.attempt_startime = time.time()
        while True:
            self.refresh_frame()
            attempt_frame = self.get_attempt_area(self.buffer_read)
            self.update_attempt()
            
            # Muestra la grabacion
            if show:
                frame = self.buffer_read
                cv2.imshow('frame',frame)
                cv2.waitKey(1) 
            
        

    def update_attempt(self, tolerance=0):
        last_frame = self.get_attempt_area(self.buffer_read)
        last_last_frame = self.get_attempt_area(self.buffer_write)

        diff = last_frame - last_last_frame
        m_norm = np.sum(abs(diff))         # Manhattan norm
        #z_norm = norm(diff.ravel(), 0)    # Zero norm
        
        if m_norm > tolerance:
            self.attempt_endtime = time.time()
            self.attempt_startime = time.time()
            self.times.append(self.attempt_endtime - self.attempt_startime)
            self.attempt += 1
            print(f"Attempt: {self.attempt}, Time: {self.times[-1]}")
            return True
        return False

    # Isolates a small area that indicates the attempt number 
    # from a frame
    def get_attempt_area(self, frame):
        return frame[0:100, 390:480]        


