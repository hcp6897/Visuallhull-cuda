import numpy as np
import cv2


img = cv2.imread('../resources/108cam1-3-5_1k/0_0.png')
# 顯示圖片

print(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):

        db = img[i][j][0]-39
        dg = img[i][j][1]-135
        dr = img[i][j][2]-39
           
        if (db**2+dg**2+dr**2)**0.5 < 1 :
           img[i][j][0]=0
           img[i][j][1]=0
           img[i][j][2]=0
       

cv2.imshow('My Image', img)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()