# importing libraries
import cv2
import numpy as np

def obtain_array(file):

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(file)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video  file")
    
    Video = []
    i=0
    # Read until video is completed
    while(cap.isOpened()):
            
        # Capture frame-by-frame
        ret, frame = cap.read()
        i+= 1
        if i ==10:
            break
        if ret == True:

            Video.append(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
        
        # When everything done, release 
        # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
    return np.array(Video)


