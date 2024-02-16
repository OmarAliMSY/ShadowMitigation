import numpy as np
from utills import FileHandler
import cv2
import datetime
import shamit.cloud_segmentation.Cloudseg as cs
import pandas as pd

df = pd.read_csv(r"SKIPPD\2017_pv_raw.csv",parse_dates=["Date"])
res = df.loc[df["Huang_E4102_kW"] >= 0]
datel =  list(res["Date"])
formatted_timestamp = [timestamp.strftime("%Y%m%d%H%M%S") for timestamp in datel]

# Create a named window with a larger size
window_name = 'Cloud Detection Result'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1600, 800)  # Adjust the size as needed

# Assuming you have already obtained paths using glob
le= ["jpg","jpeg"]
FH = FileHandler(dataset_path=r"SKIPPD",foldername="2017",legal_extensions=le,include_subdirectories=True)
filenames_without_ext = FH.files_no_extensions()
filenames = FH.path_file()


for filename, filename_without_ext in zip(filenames, filenames_without_ext):
    ts = datetime.datetime.strptime(filename_without_ext, "%Y%m%d%H%M%S")

    modified_filename = filename_without_ext[:-2] + "00"

    if filename_without_ext in formatted_timestamp or modified_filename in formatted_timestamp:
        image = cv2.imread(str(filename))
    
   
        image_resized = cv2.resize(image, (64, 64))  # Resize the original image to 64x64

        # Run cloud_detection function
        csm = cs.CloudSeg(time=ts)
        cloud_cover, cloud_mask, sun_mask = csm.cloud_detection(image_resized)
        sun_center_x, sun_center_y, sun_mask = csm.sun_position()
        canvas = np.zeros((64, 64*2, 3), dtype=np.uint8)
#   
        # Original image on the left
        canvas[:64, :64] = image_resized

        result_image = image_resized.copy()
        result_image = cv2.addWeighted(result_image, 1, sun_mask, 1, 0)
        result_image = cv2.addWeighted(result_image, 1, cloud_mask, 0.3, 0)
        canvas[0:64, 64:] = result_image

        save_im = cv2.cvtColor(cloud_mask,cv2.COLOR_BGR2GRAY)

#   
        # Result image on the right (above the original image)
        text = f'Sun position: ({sun_center_x},{sun_center_y}), Cloud fraction={cloud_cover:.2f}'
        cv2.putText(img=canvas, text=text, org=(10, 2048 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow(window_name, canvas)
        cv2.imwrite(filename=f'SKIPPD/CloudMask/{filename_without_ext}.jpg',img=save_im)
        cv2.waitKey(1)  # Wait for 1 second (1000 milliseconds)
            
        
        
       
    
    
cv2.destroyWindow(window_name)
