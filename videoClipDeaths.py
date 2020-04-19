
import cv2
import numpy as np

def get_attempt_area(frame):
    return frame[0:100, 390:480]       

def attempt_has_changed(frame1_attempt_area, frame_2_attempt_area, tolerance=30000): # !!!! TOL
    diff = frame1_attempt_area - frame_2_attempt_area
    
    m_norm = np.sum(abs(diff))         # Manhattan norm
    #z_norm = norm(diff.ravel(), 0)    # Zero norm
   
    if m_norm > tolerance:
        print("mnorm " + str(m_norm))
        return True
    return False

# Returns a list with the frames in which attempt # changes
def videoFindDeaths(vidPath):
    cap = cv2.VideoCapture(vidPath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    dimmensions = size + (3, )

    frame =  np.zeros(dimmensions, dtype=np.int8)
    last_frame =  np.zeros(dimmensions, dtype=np.int8)

    deathFrames = []
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        attempt_area_1 = get_attempt_area(frame)
        attempt_area_2 = get_attempt_area(last_frame)

        cv2.waitKey(1) 
        if attempt_has_changed(attempt_area_1, attempt_area_2):
            print("Attempt has changed "  + str(frame_number))
            deathFrames.append(frame_number)

        last_frame = frame
    return deathFrames, total_frames


def get_video_segment(videoPath, skip_len=1):
    vidPath = videoPath
   

    # Build the segments
    segRange = []
    offset = 0   
    deathFrames, total_frames = videoFindDeaths(vidPath)    
    
    print("Deathframes: " + str(deathFrames))
    for death in deathFrames:
        print("OFFSET" + str(offset))
        if (death-skip_len < offset):
            offset = death+skip_len
        else:
            segRange.append((offset, (death-skip_len)))
            offset = death+skip_len
            # If deaths overlap (offset has to be higher than 0)
            if death-skip_len < offset:
                offset = death+skip_len
    

    if offset < total_frames:
        segRange.append((offset, total_frames))

    if len(segRange) == 0:
        return None

        
    print (f"Segments: {segRange}")

    return segRange

def videoClipDeaths(videoPath, output, skip_len=1):

    vidPath = videoPath
    shotsPath = output
    
    segRange = get_video_segment(vidPath, skip_len)
    if not segRange:
        print("Empty segment range")
        return None
    print (f"Segments: {segRange}")

    # Open the video
    cap = cv2.VideoCapture(vidPath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    dimmensions = size + (3, )
    frame =  np.zeros(dimmensions, dtype=np.int8)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(shotsPath, fourcc, fps, size)
    
    # Iterate through the frames of the video from segRange
    for idx, (begFidx, endFidx) in enumerate(segRange):
        cap.set(cv2.CAP_PROP_POS_FRAMES, begFidx)
        ret = True  # has frame returned
        while (cap.isOpened() and ret and writer.isOpened()):
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
 

            # End contition
            if frame_number < endFidx:
                writer.write(frame)
            else:
                break

    writer.release()
 
    # Check how many frame that new.avi has
    cap2 = cv2.VideoCapture(shotsPath)
    print(cap2.get(cv2.CAP_PROP_FRAME_COUNT))





