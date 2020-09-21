import cv2
import numpy as np 

num_down = 2
num_bil = 7

img = cv2.imread("")
img_resize = cv2.resize(img,(600,600))


# Downsampling 
img_dwn = img_resize
for i in range(num_down):
    img_dwn = cv2.pyrDown(img_dwn)

for i in range(num_bil):
    img_dwn = cv2.bilateralFilter(img_dwn,d=9,sigmaColor=9,sigmaSpace=7)

# Upsampling
for i in range(num_down):
    img_dwn = cv2.pyrUp(img_dwn)

img_gr = cv2.cvtColor(img_resize,cv2.COLOR_RGB2GRAY)
img_blr = cv2.medianBlur(img_gr,7)
img_ed = cv2.adaptiveThreshold(img_blr,255,cv2.ADAPTIVE_THRESH_MEAN_C,blockSize=9,C=2)


img_ed = cv2.cvtColor(img_ed,cv2.COLOR_GRAY2RGB)
img_cart = cv2.bitwise_and(img_dwn,img_ed)

stack = np.hastack([img_resize,img_cart])
cv2.imshow("Cartoonized",stack)

