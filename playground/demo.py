from pathlib import Path
import cv2 as cv

im = cv.imread(r"C:\Users\o.abdulmalik\Documents\Shadow-Mitigation\SKIPPD\05\20171101060000.jpg")
save_im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)


for row in save_im:
    for col in row:
        print(col)