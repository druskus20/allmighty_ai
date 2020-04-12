from Neuroball import Neuroball 
import time
from PIL import Image


def main():
    n = Neuroball((1, 26, 720+1, 576+26))
    count = 1
    while True:
        count+=1
        n.recorder_thread(show=True)
        #img = Image.fromarray(img_array) 
        #img_array.save(f'img{count}.jpg')

    
       
if __name__ == "__main__":
    main()