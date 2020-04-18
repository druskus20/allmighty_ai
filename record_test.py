from ScreenRecorder import ScreenRecorder 
from videoClipDeaths import videoClipDeaths
import time
from PIL import Image


def main():

    n = ScreenRecorder((1, 26, 720+1, 576+26))


    n.capture_live(show=True, save=True, savePath="test.mp4")
    #img = Image.fromarray(img_array) 
    #img_array.save(f'img{count}.jpg')

    videoClipDeaths("test.mp4", "test_nodeaths.mp4", 20)
    
       
if __name__ == "__main__":
    main()