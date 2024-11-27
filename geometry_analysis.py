import cv2
import numpy as np

def find_contours_gradient(image):
    # Read image and convert to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute gradients
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize magnitude to 0-255 range
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply Canny edge detection
    edges = cv2.Canny(magnitude, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    
    return image

# importing libraries
import cv2
import numpy as np

def play_contour(file):

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(file)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video  file")
    i = 0
    # Read until video is completed
    while(cap.isOpened()):
            
        # Capture frame-by-frame
        ret, frame = cap.read()


        if ret == True:
        
            # Display the resulting frame
            cv2.imshow('Frame', find_contours_gradient(frame))
        
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




if __name__=="__main__":
    # Usage
    result_image, found_contours = find_contours_gradient('path/to/your/image.jpg')
    cv2.imshow('Contours', result_image)

    cv2.destroyAllWindows()