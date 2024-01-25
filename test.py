import numpy as np
import os
import cv2
import datetime
from glob import glob
import shamit.cloud_segmentation.Cloudseg as cs

# Create a named window with a larger size
window_name = 'Cloud Detection Result'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1600, 800)  # Adjust the size as needed

# Assuming you have already obtained paths using glob
samplepath = glob(r"C:\Users\o.abdulmalik\Documents\SUNSET\2017_03_images_raw\03\10\*.jpg")

# Now, for each path in the samplepath, we strip the directory and extension
filenames_without_ext = [os.path.splitext(os.path.basename(path))[0] for path in samplepath]

for impath, ts in zip(samplepath, filenames_without_ext):
    ts = datetime.datetime.strptime(ts, "%Y%m%d%H%M%S")

    image = cv2.imread(impath)
    image_resized = cv2.resize(image, (64, 64))  # Resize the original image to 64x64
    
    # Run cloud_detection function
    csm = cs.CloudSeg(time=ts)
    cloud_cover, cloud_mask, sun_mask = csm.cloud_detection(image_resized)
    sun_center_x, sun_center_y, sun_mask = csm.sun_position()

    # Resize the cloud mask and sun mask to the original size
    
    #image = cv2.resize(image,(resize_dim,resize_dim))
    # Create a canvas to display images side by side
    canvas = np.zeros((64, 64*2, 3), dtype=np.uint8)

    # Original image on the left
    canvas[:64, :64] = image_resized
    
    result_image = image_resized.copy()
    result_image = cv2.addWeighted(result_image, 1, sun_mask, 1, 0)
    result_image = cv2.addWeighted(result_image, 1, cloud_mask, 0.3, 0)
    

    # Result image on the right (above the original image)
    canvas[0:64, 64:] = result_image

    # Sun mask on the right (below cloud mask)
    
  

    # Add text to the canvas
    text = f'Sun position: ({sun_center_x},{sun_center_y}), Cloud fraction={cloud_cover:.2f}'
    cv2.putText(img=canvas, text=text, org=(10, 2048 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Display the canvas in the named window
    cv2.imshow(window_name, canvas)

    cv2.waitKey(1)  # Wait for 1 second (1000 milliseconds)

# Close the named window after processing all images
cv2.destroyWindow(window_name)
