from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Resmin pc'deki konumu")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
ratio = img.shape[0] / 500
org = img.copy()
img = imutils.resize(img,height = 500)

gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gri = cv2.GaussianBlur(gri,(5,5),0)
edge = cv2.Canny(gri,75,200)

kontur = cv2.findContours(edge.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
kontur = imutils.grab_contours(kontur   )
kontur = sorted(kontur, key=cv2.contourArea, reverse=True)[:5]

for i in kontur:
    peri = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02*peri,True)

    if len(approx) == 4:
        ss = approx
        break

warped = four_point_transform(org,ss.reshape(4,2)* ratio)
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
t = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > t).astype("uint8")*255


cv2.drawContours(img, [ss],-1, (0,255,0),2)
cv2.imshow("org",imutils.resize(org, height=650))
cv2.imshow("scan",imutils.resize(warped,height=650))

cv2.waitKey(0)
cv2.destroyAllWindows()
